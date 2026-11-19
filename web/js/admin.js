/**
 * AI Meeting Studio - Quản trị Hệ thống & Nhân sự (/admin)
 * 1. Xác thực quyền Quản trị viên (is_admin == 1)
 * 2. Quản lý tài khoản nhân sự, chức vụ, phân quyền và mẫu giọng nói
 */

const STORAGE_USER_KEY = "ai_meeting_current_user_v2";
const STORAGE_TOKEN_KEY = "ai_meeting_access_token_v2";

function getAccessToken() {
  return localStorage.getItem(STORAGE_TOKEN_KEY) || "";
}

function setAccessToken(token) {
  if (token) {
    localStorage.setItem(STORAGE_TOKEN_KEY, token);
  }
}

function clearAccessToken() {
  localStorage.removeItem(STORAGE_TOKEN_KEY);
}

// Tự động đính kèm Authorization: Bearer <token> cho mọi lệnh fetch() và xử lý 401
const _nativeAdminFetch = window.fetch.bind(window);
window.fetch = async function (input, init = {}) {
  const urlStr = typeof input === "string" ? input : input?.url || "";
  const token = getAccessToken();
  const headers = new Headers(init.headers || {});

  if (token && urlStr.startsWith("/api/") && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await _nativeAdminFetch(input, { ...init, headers });

  const isAuthEndpoint =
    urlStr.startsWith("/api/auth/login") || urlStr.startsWith("/api/auth/register");
  if (response.status === 401 && urlStr.startsWith("/api/") && !isAuthEndpoint) {
    adminState.currentAdmin = null;
    clearAccessToken();
    localStorage.removeItem(STORAGE_USER_KEY);
    if (typeof lockAdminPortal === "function") {
      lockAdminPortal("Phiên đăng nhập đã hết hạn hoặc không hợp lệ. Vui lòng đăng nhập lại.");
    }
  }
  return response;
};

const adminState = {
  currentAdmin: null,
  users: [],
  stats: {},
  isRecording: false,
  audioContext: null,
  mediaStream: null,
  scriptProcessor: null,
  recordedBuffers: [],
  recordingSampleRate: 44100,
  recordingSeconds: 0,
  recordingTimerId: null,
  recordedWavBlob: null,
};

const AVATAR_STYLE = "bg-slate-100 text-slate-700 border border-slate-200";

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function getInitials(name) {
  const parts = String(name || "U").trim().split(/\s+/);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return String(name || "U").slice(0, 2).toUpperCase();
}

function formatTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const el = document.createElement("div");
  const colorCls =
    type === "error"
      ? "bg-rose-600 text-white"
      : type === "info"
        ? "bg-[#0F172A] text-white"
        : "bg-[#0F172A] text-white";
  el.className = `${colorCls} px-4 py-2.5 rounded-xl shadow-lg text-xs font-semibold flex items-center gap-2 transition-all duration-200`;
  el.innerHTML = `<span>${escapeHtml(message)}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 220);
  }, 3200);
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

/* =========================================================================
   ADMIN PORTAL RBAC GATE & AUTHENTICATION
   ========================================================================= */

function showAdminAuthError(message) {
  const box = document.getElementById("admin-auth-error");
  const txt = document.getElementById("admin-auth-error-text");
  if (box && txt) {
    txt.textContent = message;
    box.classList.remove("hidden");
  }
}

function hideAdminAuthError() {
  document.getElementById("admin-auth-error")?.classList.add("hidden");
}

function lockAdminPortal(errorMsg = "") {
  document.getElementById("admin-protected-shell")?.classList.add("hidden");
  document.getElementById("admin-auth-overlay")?.classList.remove("hidden");
  if (errorMsg) {
    showAdminAuthError(errorMsg);
  } else {
    hideAdminAuthError();
  }
  refreshIcons();
}

async function unlockAdminPortal(adminUser) {
  adminState.currentAdmin = adminUser;
  localStorage.setItem(STORAGE_USER_KEY, JSON.stringify(adminUser));
  hideAdminAuthError();

  document.getElementById("admin-auth-overlay")?.classList.add("hidden");
  document.getElementById("admin-protected-shell")?.classList.remove("hidden");
  const activeLabel = document.getElementById("admin-active-user-label");
  if (activeLabel) {
    activeLabel.textContent = `${adminUser.name} (@${adminUser.username} · ${adminUser.role})`;
  }

  await loadAdminDirectory();
  refreshIcons();
}

async function initAdminPortal() {
  refreshIcons();
  let savedUser = null;
  const savedToken = getAccessToken();
  try {
    const raw = localStorage.getItem(STORAGE_USER_KEY);
    if (raw) savedUser = JSON.parse(raw);
  } catch (_) { }

  if (!savedUser || !savedUser.username || !savedToken) {
    clearAccessToken();
    localStorage.removeItem(STORAGE_USER_KEY);
    lockAdminPortal();
    return;
  }

  if (!savedUser.is_admin) {
    lockAdminPortal(
      "Truy cập bị từ chối: Tài khoản của bạn không có quyền Quản trị viên."
    );
    return;
  }

  await unlockAdminPortal(savedUser);
}

async function submitAdminSignIn() {
  hideAdminAuthError();
  const username = document.getElementById("admin-login-username")?.value.trim() || "";
  const password = document.getElementById("admin-login-password")?.value || "";

  if (!username || !password) {
    showAdminAuthError("Vui lòng nhập đầy đủ Tên đăng nhập và Mật khẩu.");
    return;
  }

  const btn = document.getElementById("btn-admin-signin");
  btn.disabled = true;

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "Tên đăng nhập hoặc mật khẩu không chính xác.");
    }

    if (!data.user || !data.user.is_admin) {
      throw new Error(
        "Truy cập bị từ chối: Tài khoản của bạn không có quyền Quản trị viên."
      );
    }

    if (data.access_token) {
      setAccessToken(data.access_token);
    }

    document.getElementById("admin-login-password").value = "";
    await unlockAdminPortal(data.user);
    showToast(`Đã xác thực quyền Quản trị viên: ${data.user.name}`);
  } catch (err) {
    showAdminAuthError(err.message);
  } finally {
    btn.disabled = false;
  }
}

function logoutAdminPortal() {
  adminState.currentAdmin = null;
  clearAccessToken();
  localStorage.removeItem(STORAGE_USER_KEY);
  lockAdminPortal();
}

/* =========================================================================
   LOAD & RENDER ADMIN DIRECTORY
   ========================================================================= */

async function loadAdminDirectory() {
  try {
    const res = await fetch("/api/admin/speakers");
    if (!res.ok) throw new Error("Không thể tải danh sách tài khoản & giọng nói");
    const data = await res.json();
    adminState.users = data.speakers || [];
    adminState.stats = data.stats || {};

    document.getElementById("admin-kpi-total").textContent =
      adminState.stats.total_users ?? adminState.users.length;
    document.getElementById("admin-kpi-verified").textContent =
      adminState.stats.verified_voiceprints ?? 0;
    document.getElementById("admin-kpi-admins").textContent =
      adminState.stats.admin_count ?? 0;
    document.getElementById("admin-kpi-storage").textContent = `${adminState.stats.storage_kb ?? 0
      } KB`;

    renderAdminTable();
    refreshIcons();
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderAdminTable() {
  const tbody = document.getElementById("admin-users-tbody");
  if (!tbody) return;

  const q = (document.getElementById("admin-search-input")?.value || "").trim().toLowerCase();
  const statusFilter = document.getElementById("admin-filter-status")?.value || "";

  const filtered = adminState.users.filter((u) => {
    const hasValidVoice = Boolean(u.has_voiceprint && u.audio_url);
    if (statusFilter === "verified" && !hasValidVoice) return false;
    if (statusFilter === "pending" && hasValidVoice) return false;
    if (!q) return true;
    const hay = `${u.id} ${u.username} ${u.name} ${u.role} ${u.department} ${u.email}`.toLowerCase();
    return hay.includes(q);
  });

  if (!filtered.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="py-10 text-center text-sm text-slate-400">
          Không tìm thấy nhân sự nào khớp với điều kiện lọc.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered
    .map((u) => {
      const roleBadge = u.is_admin
        ? `<span class="text-[10px] font-semibold px-2 py-0.5 rounded-md bg-slate-900 text-white">Quản trị viên</span>`
        : `<span class="text-[10px] font-medium px-2 py-0.5 rounded-md bg-slate-100 text-slate-600 border border-slate-200">Thành viên</span>`;

      const hasValidVoice = Boolean(u.has_voiceprint && u.audio_url);

      const vpStatus = hasValidVoice
        ? `<div class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold text-[11px]">
             <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
             Đã thu âm
           </div>`
        : `<div class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 border border-slate-200 font-semibold text-[11px]">
             <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
             Chưa thu âm
           </div>`;

      const jwtToken = encodeURIComponent(getAccessToken());
      const audioCell = hasValidVoice
        ? `<audio controls preload="none" src="${u.audio_url}?token=${jwtToken}&t=${Date.now()}" class="h-8 w-56"></audio>`
        : `<span class="text-xs italic text-slate-400">Chưa có bản ghi</span>`;

      return `
        <tr class="hover:bg-slate-50/80 transition">
          <td class="py-3.5 px-4">
            <div class="flex items-center gap-2">
              <span class="font-mono-time font-bold text-[#0F172A]">#${u.id}</span>
              ${roleBadge}
            </div>
            <div class="font-mono-time text-[11px] text-slate-500 mt-0.5">@${escapeHtml(
        u.username || "user_" + u.id
      )}</div>
          </td>

          <td class="py-3.5 px-4">
            <div class="flex items-center gap-2.5">
              <div class="w-8 h-8 rounded-full ${AVATAR_STYLE} text-xs font-bold flex items-center justify-center shrink-0">
                ${escapeHtml(getInitials(u.name))}
              </div>
              <div>
                <div class="text-xs font-bold text-[#0F172A]">${escapeHtml(u.name)}</div>
                <div class="text-[11px] text-slate-400">${escapeHtml(u.email || "Chưa cập nhật email")}</div>
              </div>
            </div>
          </td>

          <td class="py-3.5 px-4">
            <span class="inline-block text-[11px] font-medium px-2.5 py-0.5 rounded-md bg-slate-100 text-slate-700 border border-slate-200 whitespace-nowrap">
              ${escapeHtml(u.role)}
            </span>
          </td>

          <td class="py-3.5 px-4">
            ${vpStatus}
          </td>

          <td class="py-3.5 px-4">
            ${audioCell}
          </td>

          <td class="py-3.5 px-4 font-mono-time text-[11px] text-slate-500">
            ${escapeHtml((u.created_at || "").slice(0, 16))}
          </td>

          <td class="py-3.5 px-4 text-right">
            <div class="inline-flex items-center justify-end gap-1.5">
              <button
                type="button"
                onclick='openEditUserModal(${JSON.stringify(u).replace(/'/g, "&#39;")}, false)'
                class="px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-white hover:bg-slate-100 text-slate-700 border border-slate-200 transition"
                title="Sửa họ tên, chức vụ, phòng ban"
              >
                Sửa
              </button>
              <button
                type="button"
                onclick='openEditUserModal(${JSON.stringify(u).replace(/'/g, "&#39;")}, true)'
                class="px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-white hover:bg-slate-100 text-slate-700 border border-slate-200 transition"
                title="Thu âm hoặc cập nhật lại mẫu giọng nói"
              >
                Đổi giọng
              </button>
              <button
                type="button"
                onclick="deleteAdminUser(${u.id}, '${escapeHtml(u.name)}')"
                class="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition"
                title="Xóa tài khoản nhân sự"
              >
                <i data-lucide="trash-2" class="w-4 h-4"></i>
              </button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");

  refreshIcons();
}

/* =========================================================================
   CREATE / EDIT USER MODAL & BROWSER VOICE RECORDER
   ========================================================================= */

function setModalRole(role) {
  const el = document.getElementById("modal-role");
  if (el) el.value = role;
}

function openCreateUserModal() {
  document.getElementById("modal-profile-id").value = "";
  document.getElementById("modal-admin-title").textContent = "Thêm nhân sự";
  document.getElementById("modal-admin-subtitle").textContent =
    "Nhập thông tin tài khoản, chức vụ và thu âm hoặc tải lên mẫu giọng nói (.wav, .mp3).";
  document.getElementById("modal-username").value = "";
  document.getElementById("modal-email").value = "";
  document.getElementById("modal-password").value = "";
  document.getElementById("modal-name").value = "";
  document.getElementById("modal-role").value = "";
  document.getElementById("modal-is-admin").checked = false;
  clearModalVoiceAudio();

  const voiceSection = document.getElementById("admin-voice-section");
  if (voiceSection) {
    voiceSection.classList.remove("ring-2", "ring-slate-900/15", "border-slate-400");
  }

  document.getElementById("admin-user-modal").classList.remove("hidden");
  refreshIcons();
}

function openEditUserModal(user, focusRecorder = false) {
  document.getElementById("modal-profile-id").value = String(user.id);
  document.getElementById("modal-admin-title").textContent = focusRecorder
    ? `Đổi mẫu giọng nói: ${user.name}`
    : `Chỉnh sửa nhân sự #${user.id}`;
  document.getElementById("modal-admin-subtitle").textContent = focusRecorder
    ? "Thu âm trực tiếp qua Micro hoặc tải tệp âm thanh (.wav, .mp3) từ máy tính."
    : "Cập nhật họ tên, vai trò, phân quyền hoặc mẫu giọng nói mới.";

  document.getElementById("modal-username").value = user.username || "";
  document.getElementById("modal-email").value = user.email || "";
  document.getElementById("modal-password").value = "";
  document.getElementById("modal-name").value = user.name || "";
  document.getElementById("modal-role").value = user.role || "";
  document.getElementById("modal-is-admin").checked = Boolean(user.is_admin);
  clearModalVoiceAudio();

  document.getElementById("admin-user-modal").classList.remove("hidden");
  refreshIcons();

  const voiceSection = document.getElementById("admin-voice-section");
  if (voiceSection) {
    if (focusRecorder) {
      voiceSection.classList.add("ring-2", "ring-slate-900/15", "border-slate-400");
      setTimeout(() => {
        voiceSection.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 60);
    } else {
      voiceSection.classList.remove("ring-2", "ring-slate-900/15", "border-slate-400");
    }
  }
}

function closeAdminUserModal() {
  if (adminState.isRecording) {
    stopModalVoiceRecording();
  }
  document.getElementById("admin-user-modal").classList.add("hidden");
}

async function onModalAudioFileSelected(input) {
  if (!input.files || input.files.length === 0) return;
  const selectedFile = input.files[0];
  adminState.recordedWavBlob = null;

  const box = document.getElementById("modal-preview-box");
  const audio = document.getElementById("modal-preview-audio");
  const label = document.getElementById("modal-preview-label");

  if (box && audio && label) {
    box.classList.remove("hidden");
    label.textContent = `Đã chọn tệp: ${selectedFile.name} (${(
      selectedFile.size / 1024
    ).toFixed(1)} KB)`;
    audio.src = URL.createObjectURL(selectedFile);
  }

  try {
    const arrayBuf = await selectedFile.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      const decodeCtx = new AudioCtx();
      const audioBuffer = await decodeCtx.decodeAudioData(arrayBuf.slice(0));
      const monoData = audioBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(monoData, audioBuffer.sampleRate, 16000);
      adminState.recordedWavBlob = encodeWAV16Bit(downsampled, 16000);
      decodeCtx.close().catch(() => {});
    }
  } catch (_) {
    // Fallback: gui truc tiep file goc (.wav / .mp3) neu trinh duyet khong decode duoc
    adminState.recordedWavBlob = null;
  }
}

async function startModalVoiceRecording() {
  if (adminState.isRecording) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Trình duyệt không hỗ trợ Microphone. Vui lòng tải lên tệp âm thanh!", "error");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);

    adminState.mediaStream = stream;
    adminState.audioContext = audioCtx;
    adminState.scriptProcessor = processor;
    adminState.recordedBuffers = [];
    adminState.recordingSampleRate = audioCtx.sampleRate;
    adminState.isRecording = true;
    adminState.recordingSeconds = 0;

    processor.onaudioprocess = (e) => {
      if (!adminState.isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      adminState.recordedBuffers.push(new Float32Array(inputData));

      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      const pct = Math.min(100, Math.round(rms * 350));
      const bar = document.getElementById("modal-level-bar");
      if (bar) bar.style.width = `${pct}%`;
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    const btnRec = document.getElementById("btn-modal-record");
    const btnStop = document.getElementById("btn-modal-stop");
    if (btnRec) {
      btnRec.disabled = true;
      btnRec.classList.add("recording-pulse");
    }
    if (btnStop) btnStop.disabled = false;

    updateModalRecorderTimer(0);
    adminState.recordingTimerId = setInterval(() => {
      adminState.recordingSeconds += 1;
      updateModalRecorderTimer(adminState.recordingSeconds);
      if (adminState.recordingSeconds >= 30) {
        stopModalVoiceRecording();
      }
    }, 1000);
  } catch (err) {
    showToast("Không thể truy cập Microphone: " + err.message, "error");
  }
}

function updateModalRecorderTimer(sec) {
  const el = document.getElementById("modal-recorder-timer");
  if (el) el.textContent = `${formatTime(sec)} / 00:30`;
}

function stopModalVoiceRecording() {
  if (!adminState.isRecording) return;
  adminState.isRecording = false;

  if (adminState.recordingTimerId) {
    clearInterval(adminState.recordingTimerId);
    adminState.recordingTimerId = null;
  }

  try {
    if (adminState.scriptProcessor) adminState.scriptProcessor.disconnect();
    if (adminState.mediaStream) adminState.mediaStream.getTracks().forEach((t) => t.stop());
    if (adminState.audioContext) adminState.audioContext.close().catch(() => { });
  } catch (_) { }

  const btnRec = document.getElementById("btn-modal-record");
  const btnStop = document.getElementById("btn-modal-stop");
  if (btnRec) {
    btnRec.disabled = false;
    btnRec.classList.remove("recording-pulse");
  }
  if (btnStop) btnStop.disabled = true;

  const bar = document.getElementById("modal-level-bar");
  if (bar) bar.style.width = "0%";

  if (!adminState.recordedBuffers.length) return;

  const merged = mergeFloat32Buffers(adminState.recordedBuffers);
  const downsampled = downsampleBuffer(merged, adminState.recordingSampleRate, 16000);
  const wavBlob = encodeWAV16Bit(downsampled, 16000);
  adminState.recordedWavBlob = wavBlob;

  const box = document.getElementById("modal-preview-box");
  const audio = document.getElementById("modal-preview-audio");
  const label = document.getElementById("modal-preview-label");
  if (box && audio && label) {
    box.classList.remove("hidden");
    label.textContent = `Đã thu âm ${adminState.recordingSeconds} giây (${(
      wavBlob.size / 1024
    ).toFixed(1)} KB)`;
    audio.src = URL.createObjectURL(wavBlob);
  }
}

function clearModalVoiceAudio() {
  adminState.recordedWavBlob = null;
  const fileInput = document.getElementById("modal-audio-file");
  if (fileInput) fileInput.value = "";
  document.getElementById("modal-preview-box")?.classList.add("hidden");
  updateModalRecorderTimer(0);
}

function mergeFloat32Buffers(buffers) {
  let totalLen = 0;
  for (const b of buffers) totalLen += b.length;
  const result = new Float32Array(totalLen);
  let offset = 0;
  for (const b of buffers) {
    result.set(b, offset);
    offset += b.length;
  }
  return result;
}

function downsampleBuffer(buffer, inputSampleRate, outSampleRate = 16000) {
  if (outSampleRate === inputSampleRate) return buffer;
  const ratio = inputSampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function encodeWAV16Bit(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

async function submitAdminUserModal() {
  const profileId = document.getElementById("modal-profile-id").value.trim();
  const username = document.getElementById("modal-username").value.trim();
  const email = document.getElementById("modal-email").value.trim();
  const password = document.getElementById("modal-password").value;
  const name = document.getElementById("modal-name").value.trim();
  const role = document.getElementById("modal-role").value.trim();
  const department = "";
  const isAdmin = document.getElementById("modal-is-admin").checked;
  const fileInput = document.getElementById("modal-audio-file");

  if (!name || !role) {
    showToast("Vui lòng nhập đầy đủ Họ tên và Chức vụ nhân sự!", "error");
    return;
  }

  let audioBlobOrFile = adminState.recordedWavBlob;
  let filename = "staff_voice_sample.wav";
  if (!audioBlobOrFile && fileInput && fileInput.files && fileInput.files.length > 0) {
    audioBlobOrFile = fileInput.files[0];
    filename = fileInput.files[0].name;
  }

  const formData = new FormData();
  formData.append("name", name);
  formData.append("role", role);
  formData.append("department", department);
  formData.append("username", username);
  formData.append("email", email);
  if (password && password.trim()) {
    formData.append("password", password.trim());
  }
  formData.append("is_admin", String(isAdmin));
  if (audioBlobOrFile) {
    formData.append("audio", audioBlobOrFile, filename);
  }

  const btn = document.getElementById("btn-modal-save");
  btn.disabled = true;

  try {
    const isEdit = Boolean(profileId);
    const url = isEdit ? `/api/admin/speakers/${profileId}` : "/api/admin/speakers";
    const method = isEdit ? "PUT" : "POST";

    const res = await fetch(url, { method, body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Không thể lưu hồ sơ nhân sự");
    }

    closeAdminUserModal();
    showToast(
      isEdit
        ? `Đã cập nhật hồ sơ của ${name} (${role}) thành công!`
        : `Đã tạo hồ sơ nhân sự ${name} (${role}) thành công!`
    );
    await loadAdminDirectory();
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function deleteAdminUser(profileId, name) {
  if (!confirm(`Bạn có chắc chắn muốn xóa tài khoản và mẫu giọng nói của "${name}"?`)) return;
  try {
    const res = await fetch(`/api/admin/speakers/${profileId}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error("Không thể xóa hồ sơ nhân sự");
    showToast(`Đã xóa tài khoản của ${name}`);
    await loadAdminDirectory();
  } catch (err) {
    showToast(err.message, "error");
  }
}

window.addEventListener("DOMContentLoaded", initAdminPortal);
