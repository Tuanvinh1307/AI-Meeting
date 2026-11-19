"""
Module trich xuat dac trung giong noi (Voiceprint Embedding) va doi sanh Nhan su (Speaker Identification).
Su dung soundfile va numpy de tinh toan vector dac trung am hoc:
- 24 dai nang luong log-Mel (Mel-filterbank energies)
- Spectral Centroid, Spectral Rolloff (85%), Zero-Crossing Rate (ZCR)
- Fundamental Frequency (Pitch / F0 distribution qua Autocorrelation)
- Thong ke Mean & Std tren cac khung tin hieu co tieng noi (voiced frames), chuan hoa L2-norm.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _hz_to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 24,
    fmin: float = 80.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Tao ma tran loc Mel (n_mels, n_fft // 2 + 1)."""
    if fmax is None:
        fmax = min(7600.0, sample_rate / 2.0)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float64)

    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        if f_center <= f_left:
            f_center = f_left + 1
        if f_right <= f_center:
            f_right = f_center + 1

        for k in range(f_left, min(f_center, n_freqs)):
            filterbank[m - 1, k] = (k - f_left) / float(f_center - f_left)
        for k in range(f_center, min(f_right, n_freqs)):
            filterbank[m - 1, k] = (f_right - k) / float(f_right - f_center)

    return filterbank


def _load_mono_slice(
    audio_path: Union[str, Path],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Doc file am thanh, cat theo start_time/end_time (neu co) va chuyen ve mono."""
    path_obj = Path(audio_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Khong tim thay file am thanh: {path_obj}")

    info = sf.info(str(path_obj))
    sr = info.samplerate
    total_frames = info.frames

    start_frame = int(max(0.0, start_time) * sr) if start_time is not None else 0
    end_frame = int(min(total_frames, int(end_time * sr))) if end_time is not None else total_frames

    if end_frame <= start_frame:
        start_frame = 0
        end_frame = total_frames

    audio, sample_rate = sf.read(
        str(path_obj),
        start=start_frame,
        stop=end_frame,
        dtype="float64",
        always_2d=True,
    )
    mono = np.mean(audio, axis=1)
    return mono, sample_rate


def save_audio_clip(
    source_audio_path: Union[str, Path],
    target_audio_path: Union[str, Path],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Path:
    """Trich xuat doan am thanh va luu thanh file WAV mono chuan."""
    mono, sr = _load_mono_slice(source_audio_path, start_time=start_time, end_time=end_time)
    target = Path(target_audio_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    peak = np.max(np.abs(mono))
    if peak > 1e-6:
        mono = mono / peak * 0.92
    sf.write(str(target), mono, sr, subtype="PCM_16")
    return target


def _compute_embedding_from_signal(mono: np.ndarray, sample_rate: int) -> List[float]:
    """Tinh vector dac trung giong noi 60 chieu da chuan hoa L2 tu tin hieu mono."""
    if mono.size < 400:
        mono = np.pad(mono, (0, max(0, 400 - mono.size)))

    # Loai bo DC offset va chuan hoa bien do
    mono = mono - np.mean(mono)
    peak = np.max(np.abs(mono))
    if peak > 1e-7:
        mono = mono / peak

    # Pre-emphasis de lam noi bat formant tan so cao
    emphasized = np.append(mono[0], mono[1:] - 0.97 * mono[:-1])

    frame_len = max(256, int(0.025 * sample_rate))
    hop_len = max(128, int(0.010 * sample_rate))
    n_fft = 512 if frame_len <= 512 else 1024

    if len(emphasized) < frame_len:
        emphasized = np.pad(emphasized, (0, frame_len - len(emphasized)))

    num_frames = 1 + (len(emphasized) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + np.arange(num_frames)[:, None] * hop_len
    frames = emphasized[idx]
    raw_frames = mono[idx]

    # Loc cac khung co nang luong tieng noi (Voiced / Active speech frames)
    frame_rms = np.sqrt(np.mean(raw_frames ** 2, axis=1) + 1e-12)
    energy_threshold = max(np.percentile(frame_rms, 25) * 0.5, 0.015)
    active_mask = frame_rms >= energy_threshold
    if np.sum(active_mask) >= 3:
        frames = frames[active_mask]
        raw_frames = raw_frames[active_mask]

    window = np.hanning(frame_len)[None, :]
    windowed = frames * window

    mag_spec = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1))
    pow_spec = (mag_spec ** 2) / float(n_fft)

    # 1. 24 dai nang luong log-Mel
    mel_fb = _create_mel_filterbank(sample_rate, n_fft=n_fft, n_mels=24)
    mel_energy = np.dot(pow_spec, mel_fb.T)
    log_mel = np.log(mel_energy + 1e-10)
    # Chuan hoa theo trung binh tung khung de bat dac trung am sac (timbre/formants) doc lap am luong
    log_mel_norm = log_mel - np.mean(log_mel, axis=1, keepdims=True)
    log_mel_scaled = log_mel_norm / (np.std(log_mel_norm) + 1e-6)

    # 2. Spectral Centroid & Spectral Rolloff (85%)
    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)[None, :]
    mag_sum = np.sum(mag_spec, axis=1, keepdims=True) + 1e-10
    centroid = np.sum(freqs * mag_spec, axis=1) / (mag_sum[:, 0] * (sample_rate / 2.0))

    cum_energy = np.cumsum(pow_spec, axis=1)
    total_energy = cum_energy[:, -1:] + 1e-10
    rolloff_idx = np.argmax(cum_energy >= 0.85 * total_energy, axis=1)
    rolloff = freqs[0, rolloff_idx] / (sample_rate / 2.0)

    # 3. Zero-Crossing Rate (ZCR)
    signs = np.sign(raw_frames)
    signs[signs == 0] = 1
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    # 4. Fundamental Frequency (Pitch / F0 distribution) qua Autocorrelation (75Hz - 400Hz)
    min_lag = max(1, int(sample_rate / 400.0))
    max_lag = min(frame_len - 1, int(sample_rate / 75.0))
    f0_list: List[float] = []
    harmonic_ratio_list: List[float] = []
    if max_lag > min_lag:
        for rf in raw_frames:
            rf_centered = rf - np.mean(rf)
            norm0 = float(np.dot(rf_centered, rf_centered))
            if norm0 < 1e-8:
                continue
            corr = np.correlate(rf_centered, rf_centered, mode="full")[len(rf_centered) - 1 :]
            corr_slice = corr[min_lag : max_lag + 1] / (norm0 + 1e-9)
            if corr_slice.size > 0:
                peak_idx = int(np.argmax(corr_slice))
                peak_val = float(corr_slice[peak_idx])
                if peak_val > 0.25:
                    lag = min_lag + peak_idx
                    f0_hz = sample_rate / float(lag)
                    f0_list.append(f0_hz / 300.0)
                    harmonic_ratio_list.append(peak_val)

    if f0_list:
        f0_arr = np.asarray(f0_list, dtype=np.float64)
        hr_arr = np.asarray(harmonic_ratio_list, dtype=np.float64)
        pitch_features = np.array(
            [
                np.mean(f0_arr),
                np.std(f0_arr),
                np.percentile(f0_arr, 25),
                np.percentile(f0_arr, 75),
                np.mean(hr_arr),
                np.std(hr_arr),
            ],
            dtype=np.float64,
        )
    else:
        pitch_features = np.zeros(6, dtype=np.float64)

    mel_mean = np.mean(log_mel_scaled, axis=0)
    mel_std = np.std(log_mel_scaled, axis=0)

    spectral_stats = np.array(
        [
            np.mean(centroid),
            np.std(centroid),
            np.mean(rolloff),
            np.std(rolloff),
            np.mean(zcr),
            np.std(zcr),
        ],
        dtype=np.float64,
    )

    feature_vec = np.concatenate([mel_mean, mel_std, spectral_stats, pitch_features])
    norm = float(np.linalg.norm(feature_vec))
    if norm > 1e-9:
        feature_vec = feature_vec / norm

    return [round(float(x), 6) for x in feature_vec]


def extract_voiceprint_embedding(
    audio_path: Union[str, Path],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[float]:
    """
    Trich xuat vector dac trung giong noi (Voiceprint Embedding) tu file am thanh.
    Chuyen mono, chuan hoa bien do, chia khung FFT trich xuat 24 dai log-Mel,
    spectral centroid, spectral rolloff, zero-crossing rate, pitch distribution,
    lay mean va std roi chuan hoa L2-norm.
    """
    mono, sample_rate = _load_mono_slice(audio_path, start_time=start_time, end_time=end_time)
    return _compute_embedding_from_signal(mono, sample_rate)


def cosine_similarity(vec_a: Union[List[float], np.ndarray], vec_b: Union[List[float], np.ndarray]) -> float:
    """Tinh do tuong dong Cosine giua 2 vector dac trung giong noi."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def match_speakers_to_profiles(
    audio_path: Union[str, Path],
    segments: list,
    profiles: list,
    threshold: float = 0.68,
) -> Dict[str, dict]:
    """
    Gom cac doan am thanh cua tung speaker_id (SPEAKER_00, SPEAKER_01...),
    tinh vector dac trung trung binh va doi sanh Cosine Similarity voi cac SpeakerProfile trong DB
    (gan 1-1 uu tien do tuong dong cao nhat).
    Tra ve {speaker_id: {"profile_id": ..., "name": ..., "role": ..., "display_label": "Ho ten (Chuc vu)", "similarity": 0.94}}.
    """
    path_obj = Path(audio_path)
    if not path_obj.exists() or not segments or not profiles:
        return {}

    # Parse profile embeddings
    parsed_profiles: List[Tuple[Any, np.ndarray]] = []
    for prof in profiles:
        try:
            raw_json = prof.voiceprint_json if hasattr(prof, "voiceprint_json") else prof.get("voiceprint_json", "[]")
            vec = np.asarray(json.loads(raw_json), dtype=np.float64)
            if vec.size > 0:
                parsed_profiles.append((prof, vec))
        except Exception as e:
            logger.warning("Khong the parse voiceprint_json cua profile: %s", e)

    if not parsed_profiles:
        return {}

    # Gom cac doan am thanh theo speaker_id
    speaker_intervals: Dict[str, List[Tuple[float, float]]] = {}
    for seg in segments:
        sid = seg.speaker_id if hasattr(seg, "speaker_id") else seg.get("speaker_id")
        st = seg.start_time if hasattr(seg, "start_time") else seg.get("start_time")
        et = seg.end_time if hasattr(seg, "end_time") else seg.get("end_time")
        if not sid or st is None or et is None:
            continue
        if float(et) > float(st) + 0.15:
            speaker_intervals.setdefault(str(sid), []).append((float(st), float(et)))

    speaker_embeddings: Dict[str, np.ndarray] = {}
    for sid, intervals in speaker_intervals.items():
        vecs: List[np.ndarray] = []
        concat_pieces: List[np.ndarray] = []
        sr_val: Optional[int] = None
        for st, et in intervals:
            try:
                piece, sr_val = _load_mono_slice(path_obj, start_time=st, end_time=et)
                if piece.size > 0:
                    concat_pieces.append(piece)
                if et - st >= 0.8:
                    emb = _compute_embedding_from_signal(piece, sr_val)
                    vecs.append(np.asarray(emb, dtype=np.float64))
            except Exception as e:
                logger.debug("Bo qua doan am thanh loi cua %s [%.2f-%.2f]: %s", sid, st, et, e)

        if concat_pieces and sr_val is not None:
            full_speaker_signal = np.concatenate(concat_pieces)
            full_emb = np.asarray(_compute_embedding_from_signal(full_speaker_signal, sr_val), dtype=np.float64)
            vecs.append(full_emb)

        if vecs:
            avg_vec = np.mean(np.stack(vecs, axis=0), axis=0)
            norm = float(np.linalg.norm(avg_vec))
            if norm > 1e-9:
                avg_vec = avg_vec / norm
            speaker_embeddings[sid] = avg_vec

    # Tinh ma tran tuong dong (similarity, speaker_id, profile)
    candidates: List[Tuple[float, str, Any]] = []
    for sid, s_vec in speaker_embeddings.items():
        for prof, p_vec in parsed_profiles:
            sim = cosine_similarity(s_vec, p_vec)
            candidates.append((sim, sid, prof))

    # Sap xep giam dan theo do tuong dong va gan 1-1
    candidates.sort(key=lambda item: item[0], reverse=True)
    assigned_speakers = set()
    assigned_profiles = set()
    matches: Dict[str, dict] = {}

    for sim, sid, prof in candidates:
        if sim < threshold:
            continue
        prof_id = prof.id if hasattr(prof, "id") else prof.get("id")
        if sid in assigned_speakers or prof_id in assigned_profiles:
            continue

        prof_name = prof.name if hasattr(prof, "name") else prof.get("name", "")
        prof_role = prof.role if hasattr(prof, "role") else prof.get("role", "")
        prof_dept = prof.department if hasattr(prof, "department") else prof.get("department", "")
        display_label = (
            prof.display_label
            if hasattr(prof, "display_label")
            else (f"{prof_name} ({prof_role})" if prof_role else prof_name)
        )

        matches[sid] = {
            "profile_id": prof_id,
            "name": prof_name,
            "role": prof_role,
            "department": prof_dept,
            "display_label": display_label,
            "similarity": round(float(sim), 4),
        }
        assigned_speakers.add(sid)
        assigned_profiles.add(prof_id)

    return matches
