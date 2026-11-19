/**
 * Quản lý Biên bản Cuộc họp & Công việc
 */

const STORAGE_USER_KEY = "ai_meeting_current_user_v2";

const state = {
  currentUser: null,
  authMode: "login",
  currentTab: "workspace",
  meetings: [],
  currentMeetingId: null,
  currentMeeting: null,
  speakersRegistry: [],
  allTasks: [],
  audioClipEndTime: null,
  audioSpeedIndex: 0,
  audioSpeeds: [1.0, 1.25, 1.5, 0.85],
  // Onboarding Browser Voice Recorder state
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

const SPEAKER_PALETTE = [
  { bg: "bg-indigo-600", light: "bg-indigo-50", text: "text-indigo-700", border: "border-indigo-200" },
  { bg: "bg-emerald-600", light: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
  { bg: "bg-amber-600", light: "bg-amber-50", text: "text-amber-700", border: "border-amber-200" },
  { bg: "bg-rose-600", light: "bg-rose-50", text: "text-rose-700", border: "border-rose-200" },
  { bg: "bg-sky-600", light: "bg-sky-50", text: "text-sky-700", border: "border-sky-200" },
  { bg: "bg-purple-600", light: "bg-purple-50", text: "text-purple-700", border: "border-purple-200" },
];

function getSpeakerStyle(speakerId) {
  const num = parseInt(String(speakerId || "0").replace(/\D/g, ""), 10) || 0;
  return SPEAKER_PALETTE[num % SPEAKER_PALETTE.length];
}

function parseNameAndRole(displayName, matchedProfile) {
  if (matchedProfile && matchedProfile.name) {
    return {
      name: matchedProfile.name,
      role: matchedProfile.role || "",
      similarity: matchedProfile.similarity || null,
    };
  }
  const raw = String(displayName || "Người nói").trim();
  const match = raw.match(/^(.+?)\s*\((.+?)\)$/);
  if (match) {
    return { name: match[1].trim(), role: match[2].trim(), similarity: null };
  }
  return { name: raw, role: "", similarity: null };
}

function getInitials(name) {
  const parts = String(name || "N").trim().split(/\s+/);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return String(name || "N").slice(0, 2).toUpperCase();
}

function formatTime(sec) {
  if (sec === null || sec === undefined || isNaN(sec)) return "00:00";
  const total = Math.max(0, Math.floor(Number(sec)));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const el = document.createElement("div");
  const colorCls =
    type === "error"
      ? "bg-rose-600 text-white"
      : type === "info"
      ? "bg-slate-800 text-white"
      : "bg-emerald-600 text-white";
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
   NAVIGATION & INITIALIZATION (3 TABS WORKSPACE)
   ========================================================================= */

function switchTab(tab) {
  state.currentTab = tab;
  const tabs = ["workspace", "studio", "tasks"];
  tabs.forEach((t) => {
    const viewEl = document.getElementById(`view-${t}`);
    const navEl = document.getElementById(`nav-${t}`);
    if (viewEl) viewEl.classList.toggle("hidden", t !== tab);
    if (navEl) navEl.classList.toggle("active", t === tab);
  });

  if (tab === "tasks") {
    loadActionCenterTasks();
  }
  refreshIcons();
}

async function initApp() {
  setupGlobalAudioEvents();
  checkAuthAndOnboardingGate();
  refreshIcons();
}

/* =========================================================================
   REAL PASSWORD AUTHENTICATION (2 TABS) & MANDATORY VOICE ONBOARDING
   ========================================================================= */

function switchAuthMode(mode) {
  state.authMode = mode;
  hideAuthError();

  const btnLogin = document.getElementById("tab-btn-login");
  const btnReg = document.getElementById("tab-btn-register");
  const formLogin = document.getElementById("form-auth-login");
  const formReg = document.getElementById("form-auth-register");

  if (mode === "login") {
    btnLogin.className = "py-2 text-xs font-semibold rounded-lg transition bg-white text-slate-900 shadow-xs";
    btnReg.className = "py-2 text-xs font-semibold rounded-lg transition text-slate-500 hover:text-slate-800";
    formLogin.classList.remove("hidden");
    formReg.classList.add("hidden");
  } else {
    btnReg.className = "py-2 text-xs font-semibold rounded-lg transition bg-white text-slate-900 shadow-xs";
    btnLogin.className = "py-2 text-xs font-semibold rounded-lg transition text-slate-500 hover:text-slate-800";
    formReg.classList.remove("hidden");
    formLogin.classList.add("hidden");
  }
}

function showAuthError(msg) {
  const box = document.getElementById("auth-error-box");
  const txt = document.getElementById("auth-error-text");
  if (box && txt) {
    txt.textContent = msg;
    box.classList.remove("hidden");
  }
}

function hideAuthError() {
  document.getElementById("auth-error-box")?.classList.add("hidden");
}

function checkAuthAndOnboardingGate() {
  let savedUser = null;
  try {
    const raw = localStorage.getItem(STORAGE_USER_KEY);
    if (raw) savedUser = JSON.parse(raw);
  } catch (_) {}

  if (!savedUser || !savedUser.username) {
    showAuthGateOverlay();
    return;
  }

  handleAuthenticatedUser(savedUser);
}

function showAuthGateOverlay() {
  document.getElementById("auth-gate-overlay")?.classList.remove("hidden");
  document.getElementById("onboarding-gate-overlay")?.classList.add("hidden");
  hideAuthError();
  refreshIcons();
}

function showOnboardingGateOverlay(user) {
  document.getElementById("auth-gate-overlay")?.classList.add("hidden");
  document.getElementById("onboarding-gate-overlay")?.classList.remove("hidden");

  document.getElementById("onboard-username-tag").textContent = `@${user.username}`;
  const nameInput = document.getElementById("onboard-name");
  const roleInput = document.getElementById("onboard-role");

  if (nameInput) {
    nameInput.value = user.name && user.name !== user.username ? user.name : "";
  }
  if (roleInput) {
    roleInput.value =
      user.role && user.role !== "Chưa thiết lập" && user.role !== "Thành viên mới"
        ? user.role
        : "";
  }
  clearOnboardingAudio();
  refreshIcons();
}

function unlockWorkspaceForUser(user) {
  document.getElementById("auth-gate-overlay")?.classList.add("hidden");
  document.getElementById("onboarding-gate-overlay")?.classList.add("hidden");

  document.getElementById("sidebar-user-avatar").textContent = getInitials(user.name);
  document.getElementById("sidebar-user-name").textContent = user.name;
  document.getElementById("sidebar-user-role").textContent = user.role || "";

  // Chỉ tài khoản Quản trị viên mới hiển thị liên kết Quản trị hệ thống
  const adminLink = document.getElementById("sidebar-admin-link");
  if (adminLink) {
    if (user.is_admin) {
      adminLink.classList.remove("hidden");
      adminLink.classList.add("inline-flex");
    } else {
      adminLink.classList.add("hidden");
      adminLink.classList.remove("inline-flex");
    }
  }
  loadDashboard(true);
  refreshIcons();
}

function handleAuthenticatedUser(user) {
  state.currentUser = user;
  localStorage.setItem(STORAGE_USER_KEY, JSON.stringify(user));

  if (!user.onboarding_completed || !user.has_voiceprint) {
    showOnboardingGateOverlay(user);
  } else {
    unlockWorkspaceForUser(user);
  }
}

async function submitLoginForm() {
  hideAuthError();
  const username = document.getElementById("login-username-input")?.value.trim() || "";
  const password = document.getElementById("login-password-input")?.value || "";

  if (!username || !password) {
    showAuthError("Vui lòng nhập đầy đủ Tên đăng nhập và Mật khẩu.");
    return;
  }

  const btn = document.getElementById("btn-submit-login");
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
    document.getElementById("login-password-input").value = "";
    handleAuthenticatedUser(data.user);
    if (!data.requires_onboarding) {
      showToast(`Chào mừng trở lại, ${data.user.name} (${data.user.role})!`);
    }
  } catch (err) {
    showAuthError(err.message);
  } finally {
    btn.disabled = false;
  }
}

async function submitRegisterForm() {
  hideAuthError();
  const username = document.getElementById("reg-username-input")?.value.trim() || "";
  const email = document.getElementById("reg-email-input")?.value.trim() || "";
  const password = document.getElementById("reg-password-input")?.value || "";

  if (!username || !email || !password) {
    showAuthError("Vui lòng điền đầy đủ Tên đăng nhập, Email công việc và Mật khẩu.");
    return;
  }

  const btn = document.getElementById("btn-submit-register");
  btn.disabled = true;

  try {
    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "Không thể đăng ký tài khoản mới.");
    }
    document.getElementById("reg-password-input").value = "";
    handleAuthenticatedUser(data.user);
    showToast("Tạo tài khoản thành công! Vui lòng thiết lập chức vụ & thu âm mẫu giọng cá nhân.", "info");
  } catch (err) {
    showAuthError(err.message);
  } finally {
    btn.disabled = false;
  }
}

function logoutCurrentUser() {
  if (state.isRecording) {
    stopOnboardingRecording();
  }
  state.currentUser = null;
  state.meetings = [];
  state.currentMeetingId = null;
  state.currentMeeting = null;
  state.allTasks = [];
  localStorage.removeItem(STORAGE_USER_KEY);
  const audio = document.getElementById("global-audio");
  if (audio) {
    audio.pause();
    audio.removeAttribute("src");
    audio.removeAttribute("data-meeting-id");
  }
  renderSidebarMeetings();
  renderMeetingSelector();
  updateWorkspaceVisibility();
  const adminLink = document.getElementById("sidebar-admin-link");
  if (adminLink) {
    adminLink.classList.add("hidden");
    adminLink.classList.remove("inline-flex");
  }
  switchAuthMode("login");
  showAuthGateOverlay();
}

function setOnboardRole(roleName) {
  const el = document.getElementById("onboard-role");
  if (el) el.value = roleName;
}

function onSelectOnboardingFile(input) {
  if (input.files && input.files.length > 0) {
    state.recordedWavBlob = null;
    const box = document.getElementById("onboard-preview-box");
    const audio = document.getElementById("onboard-preview-audio");
    const status = document.getElementById("onboard-preview-status");
    if (box && audio && status) {
      box.classList.remove("hidden");
      status.textContent = `Đã chọn tệp mẫu giọng: ${input.files[0].name}`;
      audio.src = URL.createObjectURL(input.files[0]);
    }
  }
}

async function startOnboardingRecording() {
  if (state.isRecording) return;
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

    state.mediaStream = stream;
    state.audioContext = audioCtx;
    state.scriptProcessor = processor;
    state.recordedBuffers = [];
    state.recordingSampleRate = audioCtx.sampleRate;
    state.isRecording = true;
    state.recordingSeconds = 0;

    processor.onaudioprocess = (e) => {
      if (!state.isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      state.recordedBuffers.push(new Float32Array(inputData));

      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      const pct = Math.min(100, Math.round(rms * 350));
      const bar = document.getElementById("onboard-level-bar");
      if (bar) bar.style.width = `${pct}%`;
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    const btnRec = document.getElementById("btn-onboard-record");
    const btnStop = document.getElementById("btn-onboard-stop");
    if (btnRec) {
      btnRec.disabled = true;
      btnRec.classList.add("recording-pulse");
    }
    if (btnStop) btnStop.disabled = false;

    updateOnboardingTimer(0);
    state.recordingTimerId = setInterval(() => {
      state.recordingSeconds += 1;
      updateOnboardingTimer(state.recordingSeconds);
      if (state.recordingSeconds >= 30) {
        stopOnboardingRecording();
      }
    }, 1000);

    showToast("Đang thu âm... Vui lòng đọc đoạn văn bản gợi ý trong 10–30 giây.", "info");
  } catch (err) {
    showToast("Không thể mở Microphone: " + err.message, "error");
  }
}

function updateOnboardingTimer(sec) {
  const el = document.getElementById("onboard-recorder-timer");
  if (el) el.textContent = `${formatTime(sec)} / 00:30`;
}

function stopOnboardingRecording() {
  if (!state.isRecording) return;
  state.isRecording = false;

  if (state.recordingTimerId) {
    clearInterval(state.recordingTimerId);
    state.recordingTimerId = null;
  }

  try {
    if (state.scriptProcessor) state.scriptProcessor.disconnect();
    if (state.mediaStream) state.mediaStream.getTracks().forEach((t) => t.stop());
    if (state.audioContext) state.audioContext.close().catch(() => {});
  } catch (_) {}

  const btnRec = document.getElementById("btn-onboard-record");
  const btnStop = document.getElementById("btn-onboard-stop");
  if (btnRec) {
    btnRec.disabled = false;
    btnRec.classList.remove("recording-pulse");
  }
  if (btnStop) btnStop.disabled = true;

  const bar = document.getElementById("onboard-level-bar");
  if (bar) bar.style.width = "0%";

  if (!state.recordedBuffers.length) {
    showToast("Chưa thu nhận được âm thanh!", "error");
    return;
  }

  const merged = mergeFloat32Buffers(state.recordedBuffers);
  const downsampled = downsampleBuffer(merged, state.recordingSampleRate, 16000);
  const wavBlob = encodeWAV16Bit(downsampled, 16000);
  state.recordedWavBlob = wavBlob;

  const box = document.getElementById("onboard-preview-box");
  const audio = document.getElementById("onboard-preview-audio");
  const status = document.getElementById("onboard-preview-status");
  if (box && audio && status) {
    box.classList.remove("hidden");
    status.textContent = `Đã thu âm ${state.recordingSeconds} giây (${(
      wavBlob.size / 1024
    ).toFixed(1)} KB)`;
    audio.src = URL.createObjectURL(wavBlob);
  }
  showToast("Đã thu âm mẫu giọng nói cá nhân thành công!");
}

function clearOnboardingAudio() {
  state.recordedWavBlob = null;
  const fileInput = document.getElementById("onboard-file-input");
  if (fileInput) fileInput.value = "";
  document.getElementById("onboard-preview-box")?.classList.add("hidden");
  updateOnboardingTimer(0);
}

async function submitMandatoryOnboarding() {
  if (!state.currentUser) return;
  const name = document.getElementById("onboard-name")?.value.trim() || "";
  const role = document.getElementById("onboard-role")?.value.trim() || "";
  const department = "";
  const fileInput = document.getElementById("onboard-file-input");

  if (!name || !role) {
    showToast("Vui lòng nhập đầy đủ Họ tên và Chức vụ của bạn!", "error");
    return;
  }

  let audioBlobOrFile = state.recordedWavBlob;
  let filename = `onboard_${state.currentUser.username || "user"}_16khz.wav`;
  if (!audioBlobOrFile && fileInput && fileInput.files && fileInput.files.length > 0) {
    audioBlobOrFile = fileInput.files[0];
    filename = fileInput.files[0].name;
  }

  if (!audioBlobOrFile) {
    showToast("Vui lòng thu âm giọng nói cá nhân hoặc tải tệp âm thanh lên!", "error");
    return;
  }

  const formData = new FormData();
  if (state.currentUser.id) formData.append("profile_id", String(state.currentUser.id));
  if (state.currentUser.username) formData.append("username", state.currentUser.username);
  formData.append("name", name);
  formData.append("role", role);
  formData.append("department", department);
  formData.append("audio", audioBlobOrFile, filename);

  const btn = document.getElementById("btn-complete-onboarding");
  btn.disabled = true;

  try {
    const res = await fetch("/api/auth/onboarding", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Không thể hoàn tất thiết lập giọng nói");
    }
    const data = await res.json();
    showToast(`Hoàn tất thiết lập hồ sơ cho ${data.user.name} (${data.user.role})!`);
    handleAuthenticatedUser(data.user);
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
  }
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

/* =========================================================================
   DASHBOARD, MEETING WORKSPACE & AUDIO STUDIO PLAYER
   ========================================================================= */

function updateWorkspaceVisibility() {
  const emptyEl = document.getElementById("workspace-empty-state");
  const contentEl = document.getElementById("workspace-content-shell");
  const audioBarEl = document.getElementById("sticky-audio-bar");
  const hasMeetings = state.meetings && state.meetings.length > 0;
  if (emptyEl) emptyEl.classList.toggle("hidden", hasMeetings);
  if (contentEl) contentEl.classList.toggle("hidden", !hasMeetings);
  if (audioBarEl) audioBarEl.classList.toggle("hidden", !hasMeetings);
}

async function loadDashboard(loadFirstMeeting = false) {
  const currentUsername = state.currentUser?.username || "";
  if (!currentUsername) {
    state.meetings = [];
    state.speakersRegistry = [];
    renderSidebarMeetings();
    renderMeetingSelector();
    updateWorkspaceVisibility();
    const badge = document.getElementById("sidebar-pending-badge");
    if (badge) badge.textContent = "0";
    return;
  }

  try {
    const res = await fetch(`/api/dashboard?username=${encodeURIComponent(currentUsername)}`);
    if (!res.ok) throw new Error("Không thể tải danh sách cuộc họp");
    const data = await res.json();

    const kpis = data.kpis || {};
    const badge = document.getElementById("sidebar-pending-badge");
    if (badge) badge.textContent = kpis.pending_tasks ?? 0;

    state.meetings = data.meetings || [];
    state.speakersRegistry = data.speakers || [];

    renderSidebarMeetings();
    renderMeetingSelector();
    updateWorkspaceVisibility();

    if (loadFirstMeeting && state.meetings.length > 0) {
      const targetId =
        state.currentMeetingId && state.meetings.some((m) => Number(m.id) === Number(state.currentMeetingId))
          ? state.currentMeetingId
          : state.meetings[0].id;
      await selectMeeting(targetId);
    } else if (state.meetings.length === 0) {
      state.currentMeetingId = null;
      state.currentMeeting = null;
    }
    refreshIcons();
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderSidebarMeetings() {
  const box = document.getElementById("sidebar-meetings-list");
  if (!box) return;
  if (!state.meetings.length) {
    box.innerHTML = `<div class="px-3 py-4 text-xs text-slate-500">Chưa có cuộc họp nào.</div>`;
    return;
  }

  box.innerHTML = state.meetings
    .map((m) => {
      const active = Number(m.id) === Number(state.currentMeetingId);
      return `
        <div
          onclick="selectMeeting(${m.id}); switchTab('workspace');"
          class="px-3 py-2 rounded-lg cursor-pointer transition flex items-center justify-between gap-2 ${
            active
              ? "bg-slate-800 text-white font-semibold"
              : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
          }"
        >
          <div class="truncate text-xs">
            <span class="text-indigo-400 font-mono-time mr-1">#${m.id}</span>
            ${escapeHtml(m.title)}
          </div>
          <span class="text-[10px] font-mono-time text-slate-500 shrink-0">${escapeHtml(
            (m.meeting_date || "").slice(5)
          )}</span>
        </div>
      `;
    })
    .join("");
}

function renderMeetingSelector() {
  const sel = document.getElementById("meeting-selector");
  if (!sel) return;
  if (!state.meetings.length) {
    sel.innerHTML = `<option value="">Chưa có cuộc họp</option>`;
    return;
  }
  sel.innerHTML = state.meetings
    .map(
      (m) =>
        `<option value="${m.id}" ${Number(m.id) === Number(state.currentMeetingId) ? "selected" : ""}>
          #${m.id}: ${escapeHtml(m.title)}
        </option>`
    )
    .join("");
}

async function selectMeeting(meetingId) {
  if (!meetingId) return;
  const currentUsername = state.currentUser?.username || "";
  state.currentMeetingId = Number(meetingId);
  renderSidebarMeetings();
  renderMeetingSelector();

  try {
    const res = await fetch(
      `/api/meetings/${meetingId}?username=${encodeURIComponent(currentUsername)}`
    );
    if (!res.ok) throw new Error("Không tìm thấy cuộc họp");
    const detail = await res.json();
    state.currentMeeting = detail;

    const audio = document.getElementById("global-audio");
    const targetSrc = `${detail.audio_url}?username=${encodeURIComponent(currentUsername)}&t=${Date.now()}`;
    if (audio && audio.getAttribute("data-meeting-id") !== String(detail.id)) {
      audio.src = targetSrc;
      audio.setAttribute("data-meeting-id", String(detail.id));
      state.audioClipEndTime = null;
    }

    document.getElementById("meeting-date-badge").textContent =
      detail.meeting_date || "Hôm nay";

    renderWorkspaceSpeakersBar(detail);
    renderWorkspaceSummaryAndDecisions(detail);
    renderWorkspaceTasks(detail.tasks || []);
    renderWorkspaceTranscript(detail.segments || [], detail.voiceprint_matches || {});
    refreshIcons();
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderWorkspaceSpeakersBar(detail) {
  const bar = document.getElementById("workspace-speakers-bar");
  if (!bar) return;
  const speakers = detail.speakers || [];
  if (!speakers.length) {
    bar.innerHTML = `<span class="text-xs text-slate-400">Chưa có thông tin người tham gia</span>`;
    return;
  }

  bar.innerHTML = speakers
    .map((sp) => {
      const style = getSpeakerStyle(sp.speaker_id);
      const parsed = parseNameAndRole(sp.speaker_name, sp.matched_profile);
      const simBadge =
        parsed.similarity !== null
          ? `<span class="text-[10px] font-mono-time px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 font-semibold">
               Khớp ${Math.round(parsed.similarity * 100)}%
             </span>`
          : "";
      const roleBadge = parsed.role
        ? `<span class="text-[11px] px-1.5 py-0.5 rounded bg-white/90 border border-slate-200 text-slate-700 font-medium">${escapeHtml(
            parsed.role
          )}</span>`
        : "";

      return `
        <div
          class="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-xl border ${style.border} ${style.light} text-xs select-none"
        >
          <span class="w-5 h-5 rounded-full ${style.bg} text-white text-[10px] font-bold flex items-center justify-center">
            ${escapeHtml(getInitials(parsed.name))}
          </span>
          <span class="font-semibold text-slate-900">${escapeHtml(parsed.name)}</span>
          ${roleBadge}
          ${simBadge}
        </div>
      `;
    })
    .join("");
}

function renderWorkspaceSummaryAndDecisions(detail) {
  const summaryEl = document.getElementById("workspace-summary");
  if (summaryEl) {
    summaryEl.textContent =
      detail.summary || "Chưa có bản tóm tắt. Hãy bấm 'Tóm tắt lại'.";
  }

  const decisions = detail.key_decisions || [];
  document.getElementById("workspace-decisions-count").textContent = `${decisions.length} mục`;
  const listEl = document.getElementById("workspace-decisions");
  if (!listEl) return;

  if (!decisions.length) {
    listEl.innerHTML = `<li class="text-xs text-slate-400 italic bg-white border border-slate-200/80 rounded-xl p-3.5">Chưa có kết luận nào được ghi nhận.</li>`;
    return;
  }

  listEl.innerHTML = decisions
    .map(
      (d) => `
      <li class="flex items-start gap-2.5 bg-white border border-slate-200/90 rounded-xl p-3.5 text-xs text-slate-700 leading-relaxed shadow-2xs">
        <i data-lucide="check-circle" class="w-4 h-4 text-emerald-600 shrink-0 mt-0.5"></i>
        <span>${escapeHtml(d)}</span>
      </li>
    `
    )
    .join("");
}

function getPriorityBadge(priority) {
  const map = {
    "Khẩn cấp": "bg-rose-50 text-rose-700 border-rose-200",
    "Cao": "bg-amber-50 text-amber-700 border-amber-200",
    "Trung bình": "bg-indigo-50 text-indigo-700 border-indigo-200",
    "Thấp": "bg-slate-100 text-slate-600 border-slate-200",
  };
  const cls = map[priority] || map["Trung bình"];
  return `<span class="text-[11px] font-semibold px-2 py-0.5 rounded-md border ${cls}">${escapeHtml(
    priority || "Trung bình"
  )}</span>`;
}

function getConfirmBadge(status) {
  if (status === "Đã duyệt") {
    return `<span class="text-[11px] font-semibold px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200">Đã xác nhận</span>`;
  }
  if (status === "Đã từ chối") {
    return `<span class="text-[11px] font-semibold px-2 py-0.5 rounded-md bg-rose-50 text-rose-700 border border-rose-200">Đã từ chối</span>`;
  }
  return `<span class="text-[11px] font-semibold px-2 py-0.5 rounded-md bg-amber-50 text-amber-700 border border-amber-200">Chờ xác nhận</span>`;
}

function renderWorkspaceTasks(tasks) {
  document.getElementById("workspace-tasks-count").textContent = `${tasks.length} công việc`;
  const box = document.getElementById("workspace-tasks-list");
  if (!box) return;

  if (!tasks.length) {
    box.innerHTML = `<div class="text-xs text-slate-400 italic bg-white border border-slate-200/80 rounded-xl p-4">Chưa có công việc nào trong cuộc họp này.</div>`;
    return;
  }

  box.innerHTML = tasks
    .map((t) => {
      const hasAudioClip =
        t.source_start_time !== null &&
        t.source_start_time !== undefined &&
        t.source_end_time !== null &&
        t.source_end_time !== undefined;
      const clipBtn = hasAudioClip
        ? `<button
             type="button"
             onclick="playAudioSegment(${Number(t.source_start_time)}, ${Number(
            t.source_end_time
          )}, 'Nghe đoạn giao việc: ${escapeHtml(t.assignee)}')"
             class="inline-flex items-center gap-1 font-mono-time text-[11px] font-semibold px-2 py-0.5 rounded bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200/80 transition"
             title="Bấm để nghe đoạn ghi âm giao việc"
           >
             <i data-lucide="play-circle" class="w-3.5 h-3.5"></i>
             [${formatTime(t.source_start_time)} - ${formatTime(t.source_end_time)}]
           </button>`
        : "";

      const urgentBadge =
        t.is_urgent && t.status !== "Hoàn thành"
          ? `<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-600 text-white ml-1.5">Gấp</span>`
          : "";

      return `
        <div class="bg-white border border-slate-200/90 rounded-xl p-4 space-y-2.5 shadow-2xs hover:border-slate-300 transition">
          <div class="flex items-start justify-between gap-2">
            <h4 class="text-sm font-semibold text-slate-900 leading-snug">${escapeHtml(t.title)}</h4>
            <div class="flex items-center gap-1.5 shrink-0">
              ${getPriorityBadge(t.priority)}
              ${getConfirmBadge(t.confirmation_status)}
            </div>
          </div>

          ${
            t.description
              ? `<p class="text-xs text-slate-600 leading-relaxed">${escapeHtml(t.description)}</p>`
              : ""
          }

          <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600 pt-1">
            <div><span class="text-slate-400">Giao việc:</span> <strong class="text-slate-800">${escapeHtml(
              t.assigner
            )}</strong></div>
            <div><span class="text-slate-400">Thực hiện:</span> <strong class="text-indigo-700">${escapeHtml(
              t.assignee
            )}</strong></div>
            <div><span class="text-slate-400">Hạn chót:</span> <span class="font-mono-time font-medium">${escapeHtml(
              t.due_date || "Chưa ấn định"
            )}</span>${urgentBadge}</div>
          </div>

          ${
            t.source_quote
              ? `<div class="bg-slate-50 border-l-2 border-indigo-400 pl-3 pr-2 py-1.5 rounded-r-lg text-xs text-slate-600 italic flex items-center justify-between gap-2">
                   <span class="truncate">"${escapeHtml(t.source_quote)}"</span>
                   ${clipBtn}
                 </div>`
              : clipBtn
          }

          <div class="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100">
            <div class="flex items-center gap-1.5">
              <button
                type="button"
                onclick="quickPatchTask(${t.id}, { confirmation_status: 'Đã duyệt' })"
                class="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 transition"
              >
                Xác nhận
              </button>
              <button
                type="button"
                onclick="quickPatchTask(${t.id}, { confirmation_status: 'Đã từ chối' })"
                class="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 transition"
              >
                Từ chối
              </button>
            </div>

            <select
              onchange="quickPatchTask(${t.id}, { status: this.value })"
              class="text-[11px] font-semibold bg-slate-50 border border-slate-200 rounded-md px-2 py-1 text-slate-700"
            >
              <option value="Cần làm" ${t.status === "Cần làm" ? "selected" : ""}>Cần làm</option>
              <option value="Đang làm" ${t.status === "Đang làm" ? "selected" : ""}>Đang làm</option>
              <option value="Hoàn thành" ${t.status === "Hoàn thành" ? "selected" : ""}>Hoàn thành</option>
            </select>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderWorkspaceTranscript(segments, voiceprintMatches) {
  const stream = document.getElementById("workspace-transcript-stream");
  if (!stream) return;

  if (!segments.length) {
    stream.innerHTML = `<div class="text-sm text-slate-400 py-8 text-center">Chưa có dữ liệu câu thoại cho cuộc họp này.</div>`;
    return;
  }

  stream.innerHTML = segments
    .map((seg, idx) => {
      const sid = seg.speaker_id || "SPEAKER_00";
      const style = getSpeakerStyle(sid);
      const matchInfo = voiceprintMatches ? voiceprintMatches[sid] : null;
      const parsed = parseNameAndRole(seg.speaker_name, matchInfo);

      const st = seg.start_time !== null && seg.start_time !== undefined ? Number(seg.start_time) : 0;
      const et = seg.end_time !== null && seg.end_time !== undefined ? Number(seg.end_time) : st + 4;

      const roleBadge = parsed.role
        ? `<span class="text-[11px] font-semibold px-2 py-0.5 rounded-md ${style.light} ${style.text} border ${style.border}">
             ${escapeHtml(parsed.role)}
           </span>`
        : "";

      return `
        <div
          id="utterance-${idx}"
          data-start="${st}"
          data-end="${et}"
          data-search="${escapeHtml((seg.speaker_name + " " + seg.text).toLowerCase())}"
          onclick="playAudioSegment(${st}, ${et}, '${escapeHtml(parsed.name)} (${formatTime(st)})', ${idx})"
          class="utterance-row group flex items-start gap-3.5"
        >
          <div class="w-9 h-9 rounded-full ${style.bg} text-white text-xs font-bold flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
            ${escapeHtml(getInitials(parsed.name))}
          </div>

          <div class="flex-1 min-w-0">
            <div class="flex flex-wrap items-center justify-between gap-2 mb-1">
              <div class="flex items-center gap-2 flex-wrap">
                <span class="text-sm font-bold text-slate-900">${escapeHtml(parsed.name)}</span>
                ${roleBadge}
              </div>

              <span class="font-mono-time text-xs font-medium px-2 py-0.5 rounded bg-slate-100 group-hover:bg-indigo-100 group-hover:text-indigo-700 text-slate-600 transition">
                ${formatTime(st)} - ${formatTime(et)}
              </span>
            </div>

            <p class="utterance-text text-sm text-slate-700 leading-relaxed">
              ${escapeHtml(seg.text)}
            </p>
          </div>
        </div>
      `;
    })
    .join("");
}

function filterTranscriptStream() {
  const q = (document.getElementById("transcript-search")?.value || "").trim().toLowerCase();
  const rows = document.querySelectorAll("#workspace-transcript-stream .utterance-row");
  rows.forEach((row) => {
    const hay = row.getAttribute("data-search") || "";
    row.classList.toggle("hidden", q.length > 0 && !hay.includes(q));
  });
}

/* =========================================================================
   STICKY AUDIO STUDIO PLAYER & SEGMENT SYNCHRONIZATION
   ========================================================================= */

function setupGlobalAudioEvents() {
  const audio = document.getElementById("global-audio");
  const scrubber = document.getElementById("audio-scrubber");
  if (!audio) return;

  audio.addEventListener("loadedmetadata", () => {
    document.getElementById("audio-total-time").textContent = formatTime(audio.duration);
    if (scrubber) scrubber.max = String(audio.duration || 100);
  });

  audio.addEventListener("timeupdate", () => {
    const cur = audio.currentTime;
    document.getElementById("audio-current-time").textContent = formatTime(cur);
    if (scrubber && !scrubber.matches(":active")) {
      scrubber.value = String(cur);
    }

    if (state.audioClipEndTime !== null && cur >= state.audioClipEndTime) {
      audio.pause();
      state.audioClipEndTime = null;
      updatePlayIcon(false);
    }

    highlightActiveUtteranceByTime(cur, !audio.paused);
  });

  audio.addEventListener("play", () => updatePlayIcon(true));
  audio.addEventListener("pause", () => updatePlayIcon(false));
  audio.addEventListener("ended", () => {
    state.audioClipEndTime = null;
    updatePlayIcon(false);
    highlightActiveUtteranceByTime(-1, false);
  });
}

function updatePlayIcon(isPlaying) {
  const btn = document.getElementById("btn-audio-toggle");
  if (!btn) return;
  btn.innerHTML = isPlaying
    ? `<i data-lucide="pause" class="w-4 h-4"></i>`
    : `<i data-lucide="play" class="w-4 h-4 ml-0.5"></i>`;
  refreshIcons();
}

function toggleAudioPlayback() {
  const audio = document.getElementById("global-audio");
  if (!audio || !audio.src) return;
  state.audioClipEndTime = null;
  if (audio.paused) {
    audio.play().catch(() => {});
  } else {
    audio.pause();
  }
}

function seekAudioFromScrubber(val) {
  const audio = document.getElementById("global-audio");
  if (!audio) return;
  state.audioClipEndTime = null;
  audio.currentTime = Number(val);
}

function cycleAudioSpeed() {
  const audio = document.getElementById("global-audio");
  state.audioSpeedIndex = (state.audioSpeedIndex + 1) % state.audioSpeeds.length;
  const spd = state.audioSpeeds[state.audioSpeedIndex];
  if (audio) audio.playbackRate = spd;
  document.getElementById("btn-audio-speed").textContent = `${spd}x`;
}

function playAudioSegment(startTime, endTime, label = "", rowIdx = null) {
  const audio = document.getElementById("global-audio");
  if (!audio) return;

  const st = Math.max(0, Number(startTime) || 0);
  const et = Number(endTime) > st ? Number(endTime) : null;
  state.audioClipEndTime = et;

  const labelEl = document.getElementById("audio-segment-label");
  if (labelEl) {
    labelEl.textContent = label
      ? `Đang phát: ${label} [${formatTime(st)} - ${formatTime(et || st + 5)}]`
      : `Phân đoạn [${formatTime(st)} - ${formatTime(et || st + 5)}]`;
  }

  audio.currentTime = st;
  audio.play().catch((err) => {
    showToast("Không thể phát âm thanh: " + err.message, "error");
  });

  if (rowIdx !== null) {
    const row = document.getElementById(`utterance-${rowIdx}`);
    if (row) {
      row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }
}

function highlightActiveUtteranceByTime(currentTime, isPlaying) {
  const rows = document.querySelectorAll("#workspace-transcript-stream .utterance-row");
  rows.forEach((row) => {
    if (!isPlaying || currentTime < 0) {
      row.classList.remove("playing");
      return;
    }
    const st = Number(row.getAttribute("data-start") || 0);
    const et = Number(row.getAttribute("data-end") || 0);
    const active = currentTime >= st - 0.15 && currentTime <= et + 0.15;
    row.classList.toggle("playing", active);
  });
}

/* =========================================================================
   TÓM TẮT LẠI / TẢI BIÊN BẢN / XÓA CUỘC HỌP
   ========================================================================= */

async function reanalyzeCurrentMeeting() {
  if (!state.currentMeetingId) return;
  const currentUsername = state.currentUser?.username || "";
  const btn = document.getElementById("btn-reanalyze");
  const oldHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i><span>Đang tóm tắt lại...</span>`;
  refreshIcons();

  try {
    const res = await fetch(
      `/api/meetings/${state.currentMeetingId}/analyze?username=${encodeURIComponent(currentUsername)}`,
      { method: "POST" }
    );
    if (!res.ok) throw new Error("Không thể tóm tắt lại biên bản");
    showToast("Đã cập nhật bản tóm tắt và danh sách công việc!");
    await loadDashboard(false);
    await selectMeeting(state.currentMeetingId);
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = oldHtml;
    refreshIcons();
  }
}

function exportMeetingMarkdown() {
  const m = state.currentMeeting;
  if (!m) return;
  const lines = [
    `# BIÊN BẢN CUỘC HỌP: ${m.title}`,
    `- **Mã hồ sơ:** #${m.id}`,
    `- **Ngày họp:** ${m.meeting_date || ""}`,
    ``,
    `## 1. Tóm tắt nội dung`,
    m.summary || "",
    ``,
    `## 2. Kết luận chính`,
    ...(m.key_decisions || []).map((d) => `- ${d}`),
    ``,
    `## 3. Công việc được giao`,
    ...(m.tasks || []).map(
      (t) =>
        `- **[${t.status}] ${t.title}** - Người nhận: **${t.assignee}** (Người giao: ${t.assigner}, Hạn: ${
          t.due_date || "Chưa ấn định"
        }, Ưu tiên: ${t.priority})`
    ),
    ``,
    `## 4. Nội dung cuộc họp`,
    ...(m.segments || []).map(
      (s) =>
        `- **[${formatTime(s.start_time)} - ${formatTime(s.end_time)}] ${s.speaker_name}:** ${s.text}`
    ),
  ];

  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `Bien_ban_hop_${m.id}.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Đã tải xuống biên bản cuộc họp");
}

async function deleteCurrentMeeting() {
  if (!state.currentMeetingId) return;
  if (!confirm(`Bạn có chắc chắn muốn xóa cuộc họp #${state.currentMeetingId}?`)) return;
  const currentUsername = state.currentUser?.username || "";

  try {
    const res = await fetch(
      `/api/meetings/${state.currentMeetingId}?username=${encodeURIComponent(currentUsername)}`,
      { method: "DELETE" }
    );
    if (!res.ok) throw new Error("Không thể xóa cuộc họp");
    showToast(`Đã xóa cuộc họp #${state.currentMeetingId}`);
    state.currentMeetingId = null;
    await loadDashboard(true);
  } catch (err) {
    showToast(err.message, "error");
  }
}

/* =========================================================================
   TẢI LÊN GHI ÂM CUỘC HỌP
   ========================================================================= */

function onSelectUploadAudio(input) {
  const file = input.files && input.files[0];
  const label = document.getElementById("upload-audio-filename");
  if (file && label) {
    label.textContent = `Đã chọn: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  }
}

async function submitMeetingUpload() {
  const title = document.getElementById("upload-title").value.trim() || "Cuộc họp mới";
  const audioInput = document.getElementById("upload-audio-file");
  const diarInput = document.getElementById("upload-diarization-file");
  const useSample = document.getElementById("upload-use-sample").checked;
  const autoAnalyze = document.getElementById("upload-auto-analyze").checked;

  const hasAudioFile = audioInput.files && audioInput.files.length > 0;
  if (!hasAudioFile && !useSample) {
    showToast("Vui lòng chọn file ghi âm .wav hoặc tích chọn dùng file mẫu!", "error");
    return;
  }

  const currentUsername = state.currentUser?.username || "";
  const formData = new FormData();
  formData.append("title", title);
  formData.append("username", currentUsername);
  formData.append("auto_analyze", String(autoAnalyze));
  formData.append("use_sample_audio", String(useSample || !hasAudioFile));
  if (hasAudioFile) {
    formData.append("audio", audioInput.files[0]);
  }
  if (diarInput.files && diarInput.files.length > 0) {
    formData.append("diarization", diarInput.files[0]);
  }

  const btn = document.getElementById("btn-run-studio");
  const progBox = document.getElementById("studio-progress-box");
  btn.disabled = true;
  progBox.classList.remove("hidden");

  try {
    const res = await fetch(
      `/api/meetings/upload?username=${encodeURIComponent(currentUsername)}`,
      {
        method: "POST",
        headers: { "X-Username": currentUsername },
        body: formData,
      }
    );
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Không thể xử lý file ghi âm cuộc họp");
    }
    const createdMeeting = await res.json();
    showToast(`Đã xử lý xong biên bản cuộc họp #${createdMeeting.id}!`);
    state.currentMeetingId = createdMeeting.id;
    await loadDashboard(false);
    await selectMeeting(createdMeeting.id);
    switchTab("workspace");
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
    progBox.classList.add("hidden");
  }
}

/* =========================================================================
   DANH SÁCH CÔNG VIỆC
   ========================================================================= */

async function loadActionCenterTasks() {
  const currentUsername = state.currentUser?.username || "";
  const box = document.getElementById("action-center-list");
  if (!currentUsername) {
    state.allTasks = [];
    if (box) {
      box.innerHTML = `<div class="bg-white border border-slate-200 rounded-xl p-8 text-center text-sm text-slate-400">Vui lòng đăng nhập để xem công việc.</div>`;
    }
    return;
  }

  const conf = document.getElementById("filter-confirmation")?.value || "";
  const st = document.getElementById("filter-status")?.value || "";
  const params = new URLSearchParams();
  params.set("username", currentUsername);
  if (conf) params.set("confirmation_status", conf);
  if (st) params.set("status", st);

  try {
    const res = await fetch(`/api/tasks?${params.toString()}`);
    if (!res.ok) throw new Error("Không thể tải danh sách công việc");
    const data = await res.json();
    state.allTasks = data.tasks || [];
    renderActionCenterTasks();
    refreshIcons();
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderActionCenterTasks() {
  const box = document.getElementById("action-center-list");
  if (!box) return;

  if (!state.allTasks.length) {
    box.innerHTML = `<div class="bg-white border border-slate-200 rounded-xl p-8 text-center text-sm text-slate-400">Chưa có công việc nào phù hợp.</div>`;
    return;
  }

  box.innerHTML = state.allTasks
    .map((t) => {
      const hasClip =
        t.source_start_time !== null &&
        t.source_start_time !== undefined &&
        t.source_end_time !== null &&
        t.source_end_time !== undefined;

      return `
        <div class="bg-white border border-slate-200 rounded-xl p-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4 shadow-2xs">
          <div class="space-y-1.5 flex-1">
            <div class="flex flex-wrap items-center gap-2">
              <span class="text-xs font-mono-time text-indigo-600 font-semibold">#${t.id}</span>
              <span class="text-xs text-slate-400">·</span>
              <button
                type="button"
                onclick="selectMeeting(${t.meeting_id}); switchTab('workspace');"
                class="text-xs font-medium text-slate-600 hover:text-indigo-600 hover:underline"
              >
                ${escapeHtml(t.meeting_title || "Cuộc họp #" + t.meeting_id)}
              </button>
              ${getPriorityBadge(t.priority)}
              ${getConfirmBadge(t.confirmation_status)}
            </div>

            <div class="text-sm font-bold text-slate-900">${escapeHtml(t.title)}</div>
            ${
              t.description
                ? `<div class="text-xs text-slate-600">${escapeHtml(t.description)}</div>`
                : ""
            }

            <div class="flex flex-wrap items-center gap-4 text-xs text-slate-600 pt-1">
              <span>Người giao: <strong>${escapeHtml(t.assigner)}</strong></span>
              <span>Người nhận: <strong class="text-indigo-700">${escapeHtml(t.assignee)}</strong></span>
              <span>Hạn chót: <strong class="font-mono-time">${escapeHtml(
                t.due_date || "Chưa ấn định"
              )}</strong>${
                t.is_urgent && t.status !== "Hoàn thành"
                  ? `<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-600 text-white ml-1.5">Gấp</span>`
                  : ""
              }</span>
            </div>

            ${
              t.source_quote
                ? `<div class="text-xs italic text-slate-500 bg-slate-50 px-3 py-1.5 rounded-lg border-l-2 border-indigo-400 mt-1">
                     "${escapeHtml(t.source_quote)}"
                   </div>`
                : ""
            }
          </div>

          <div class="flex flex-wrap items-center gap-2 shrink-0">
            ${
              hasClip
                ? `<button
                     type="button"
                     onclick="playTaskEvidenceFromActionCenter(${t.meeting_id}, ${Number(
                    t.source_start_time
                  )}, ${Number(t.source_end_time)}, '${escapeHtml(t.assignee)}')"
                     class="px-3 py-1.5 rounded-lg text-xs font-mono-time font-semibold bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 flex items-center gap-1.5"
                     title="Nghe đoạn giao việc"
                   >
                     <i data-lucide="volume-2" class="w-3.5 h-3.5"></i>
                     Nghe đoạn giao việc [${formatTime(t.source_start_time)} - ${formatTime(t.source_end_time)}]
                   </button>`
                : ""
            }

            <button
              type="button"
              onclick="quickPatchTask(${t.id}, { confirmation_status: 'Đã duyệt' })"
              class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200"
            >
              Xác nhận
            </button>
            <button
              type="button"
              onclick="quickPatchTask(${t.id}, { confirmation_status: 'Đã từ chối' })"
              class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200"
            >
              Từ chối
            </button>
            <select
              onchange="quickPatchTask(${t.id}, { status: this.value })"
              class="text-xs font-semibold bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-700"
            >
              <option value="Cần làm" ${t.status === "Cần làm" ? "selected" : ""}>Cần làm</option>
              <option value="Đang làm" ${t.status === "Đang làm" ? "selected" : ""}>Đang làm</option>
              <option value="Hoàn thành" ${t.status === "Hoàn thành" ? "selected" : ""}>Hoàn thành</option>
            </select>
          </div>
        </div>
      `;
    })
    .join("");
}

async function playTaskEvidenceFromActionCenter(meetingId, startTime, endTime, assignee) {
  if (Number(state.currentMeetingId) !== Number(meetingId)) {
    await selectMeeting(meetingId);
  }
  playAudioSegment(startTime, endTime, `Nghe đoạn giao việc: ${assignee}`);
}

async function quickPatchTask(taskId, payload) {
  try {
    const res = await fetch(`/api/tasks/${taskId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Không thể cập nhật công việc");
    showToast("Đã cập nhật trạng thái công việc!");
    await loadDashboard(false);
    if (state.currentTab === "tasks") {
      await loadActionCenterTasks();
    }
    if (state.currentMeetingId) {
      await selectMeeting(state.currentMeetingId);
    }
  } catch (err) {
    showToast(err.message, "error");
  }
}

window.addEventListener("DOMContentLoaded", initApp);
