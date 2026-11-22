"""
Module trich xuat dac trung giong noi (Voiceprint Embedding) va doi sanh Nhan su (Speaker Identification).
Duong dan: src/core/ai/diarization/voiceprint.py
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

    mono = mono - np.mean(mono)
    peak = np.max(np.abs(mono))
    if peak > 1e-7:
        mono = mono / peak

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

    mel_fb = _create_mel_filterbank(sample_rate, n_fft=n_fft, n_mels=24)
    mel_energy = np.dot(pow_spec, mel_fb.T)
    log_mel = np.log(mel_energy + 1e-10)
    log_mel_norm = log_mel - np.mean(log_mel, axis=1, keepdims=True)
    log_mel_scaled = log_mel_norm / (np.std(log_mel_norm) + 1e-6)

    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)[None, :]
    mag_sum = np.sum(mag_spec, axis=1, keepdims=True) + 1e-10
    centroid = np.sum(freqs * mag_spec, axis=1) / (mag_sum[:, 0] * (sample_rate / 2.0))

    cum_energy = np.cumsum(pow_spec, axis=1)
    total_energy = cum_energy[:, -1:] + 1e-10
    rolloff_idx = np.argmax(cum_energy >= 0.85 * total_energy, axis=1)
    rolloff = freqs[0, rolloff_idx] / (sample_rate / 2.0)

    signs = np.sign(raw_frames)
    signs[signs == 0] = 1
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

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
    """Trich xuat vector dac trung giong noi (Voiceprint Embedding) tu file am thanh."""
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
    """Gom cac doan am thanh cua tung speaker_id va doi sanh Cosine Similarity voi SpeakerProfile trong DB."""
    path_obj = Path(audio_path)
    if not path_obj.exists() or not segments or not profiles:
        return {}

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
        # Toi uu toc do: chi lay toi da 3 doan noi dai va ro nhat (>= 0.8s, hoac top 3 doan dai nhat)
        # cho moi cum speaker_id thay vi trich xuat embedding tren toan bo cac cau le
        clear_intervals = [iv for iv in intervals if (iv[1] - iv[0]) >= 0.8]
        candidate_intervals = clear_intervals if clear_intervals else intervals
        top_intervals = sorted(
            candidate_intervals,
            key=lambda iv: iv[1] - iv[0],
            reverse=True,
        )[:3]

        vecs: List[np.ndarray] = []
        concat_pieces: List[np.ndarray] = []
        sr_val: Optional[int] = None
        for st, et in top_intervals:
            try:
                # Gioi han moi doan mau toi da 15s de tinh autocorrelation F0 nhanh va on dinh
                clipped_et = min(et, st + 15.0)
                piece, sr_val = _load_mono_slice(path_obj, start_time=st, end_time=clipped_et)
                if piece.size > 0:
                    concat_pieces.append(piece)
                    emb = _compute_embedding_from_signal(piece, sr_val)
                    vecs.append(np.asarray(emb, dtype=np.float64))
            except Exception as e:
                logger.debug("Bo qua doan am thanh loi cua %s [%.2f-%.2f]: %s", sid, st, et, e)

        if len(concat_pieces) > 1 and sr_val is not None:
            full_speaker_signal = np.concatenate(concat_pieces)
            full_emb = np.asarray(_compute_embedding_from_signal(full_speaker_signal, sr_val), dtype=np.float64)
            vecs.append(full_emb)

        if vecs:
            avg_vec = np.mean(np.stack(vecs, axis=0), axis=0)
            norm = float(np.linalg.norm(avg_vec))
            if norm > 1e-9:
                avg_vec = avg_vec / norm
            speaker_embeddings[sid] = avg_vec

    candidates: List[Tuple[float, str, Any]] = []
    for sid, s_vec in speaker_embeddings.items():
        for prof, p_vec in parsed_profiles:
            sim = cosine_similarity(s_vec, p_vec)
            candidates.append((sim, sid, prof))

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


def _extract_discriminative_subvector(vec: np.ndarray) -> np.ndarray:
    """Trich xuat vector con nhay voi am sac va cao do giong noi (Pitch F0)."""
    if vec.size < 60:
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-9 else vec
    sub = np.concatenate([vec[0:24], vec[48:54] * 1.2, vec[54:60] * 2.0])
    norm = float(np.linalg.norm(sub))
    return sub / norm if norm > 1e-9 else sub


def auto_diarize_and_identify_segments(
    audio_path: Union[str, Path],
    segments: List[Any],
    profiles: Optional[List[Any]] = None,
    profile_threshold: float = 0.72,
    cluster_threshold: float = 0.86,
    max_merge_gap: float = 2.0,
    max_merged_duration: float = 20.0,
) -> List[Any]:
    """
    Tu dong phan vai nguoi noi cho danh sach segments khi KHONG tai kem file diarization:
    - Doi chieu cosine similarity voi danh sach giong mau nhan su trong PostgreSQL.
    - Voi cac doan khong khop DB, gom cum (clustering) thanh 'Người nói 1', 'Người nói 2'...
    - Gop cac cau lien tiep cua cung mot nguoi noi.
    """
    if not segments:
        return []

    path_obj = Path(audio_path) if audio_path else None
    if path_obj is None or not path_obj.exists():
        for idx, seg in enumerate(segments):
            if not seg.speaker_id or seg.speaker_id == "SPEAKER_UNASSIGNED":
                seg.speaker_id = f"SPEAKER_{idx % 2:02d}"
                seg.speaker_name = f"Người nói {(idx % 2) + 1}"
        return segments

    parsed_profiles: List[Tuple[Any, np.ndarray, str]] = []
    for prof in profiles or []:
        try:
            raw_json = (
                prof.voiceprint_json
                if hasattr(prof, "voiceprint_json")
                else prof.get("voiceprint_json", "[]")
            )
            vec = np.asarray(json.loads(raw_json or "[]"), dtype=np.float64)
            if vec.size == 0:
                continue
            prof_name = prof.name if hasattr(prof, "name") else prof.get("name", "")
            prof_role = prof.role if hasattr(prof, "role") else prof.get("role", "")
            display_label = (
                prof.display_label
                if hasattr(prof, "display_label")
                else (f"{prof_name} ({prof_role})" if prof_role else prof_name)
            )
            parsed_profiles.append((prof, vec, display_label))
        except Exception as exc:
            logger.debug("Bo qua profile voiceprint khong hop le: %s", exc)

    seg_vectors: List[Optional[np.ndarray]] = []
    seg_disc_vectors: List[Optional[np.ndarray]] = []

    for seg in segments:
        st = getattr(seg, "start_time", None)
        et = getattr(seg, "end_time", None)
        try:
            mono, sr = _load_mono_slice(path_obj, start_time=st, end_time=et)
            if mono.size > 0:
                emb = np.asarray(_compute_embedding_from_signal(mono, sr), dtype=np.float64)
                seg_vectors.append(emb)
                seg_disc_vectors.append(_extract_discriminative_subvector(emb))
            else:
                seg_vectors.append(None)
                seg_disc_vectors.append(None)
        except Exception as exc:
            logger.debug("Khong the trich xuat vector cho segment [%.2f-%.2f]: %s", st or 0, et or 0, exc)
            seg_vectors.append(None)
            seg_disc_vectors.append(None)

    profile_speaker_ids: Dict[str, str] = {}
    next_speaker_slot = 0
    unmatched_indices: List[int] = []

    for idx, seg in enumerate(segments):
        vec = seg_vectors[idx]
        if vec is None:
            unmatched_indices.append(idx)
            continue

        best_label: Optional[str] = None
        best_sim = -1.0
        for _prof, p_vec, display_label in parsed_profiles:
            sim = cosine_similarity(vec, p_vec)
            if sim > best_sim:
                best_sim = sim
                best_label = display_label

        if best_label is not None and best_sim >= profile_threshold:
            if best_label not in profile_speaker_ids:
                profile_speaker_ids[best_label] = f"SPEAKER_{next_speaker_slot:02d}"
                next_speaker_slot += 1
            seg.speaker_id = profile_speaker_ids[best_label]
            seg.speaker_name = best_label
        else:
            unmatched_indices.append(idx)

    if unmatched_indices:
        clusters: List[Dict[str, Any]] = []

        for idx in unmatched_indices:
            vec = seg_vectors[idx]
            disc_vec = seg_disc_vectors[idx]
            if vec is None or disc_vec is None:
                if not clusters:
                    c_num = 1
                    c_sid = f"SPEAKER_{next_speaker_slot:02d}"
                    next_speaker_slot += 1
                    clusters.append(
                        {
                            "speaker_id": c_sid,
                            "speaker_name": f"Người nói {c_num}",
                            "centroid": np.zeros(60, dtype=np.float64),
                            "disc_centroid": np.zeros(36, dtype=np.float64),
                            "count": 1,
                            "members": [idx],
                        }
                    )
                else:
                    clusters[-1]["members"].append(idx)
                segments[idx].speaker_id = clusters[-1]["speaker_id"]
                segments[idx].speaker_name = clusters[-1]["speaker_name"]
                continue

            best_c_idx = -1
            best_combined_sim = -1.0
            for c_idx, c_info in enumerate(clusters):
                if np.linalg.norm(c_info["centroid"]) > 1e-6:
                    raw_sim = cosine_similarity(vec, c_info["centroid"])
                    d_sim = cosine_similarity(disc_vec, c_info["disc_centroid"])
                    combined_sim = 0.45 * raw_sim + 0.55 * d_sim
                    if combined_sim > best_combined_sim:
                        best_combined_sim = combined_sim
                        best_c_idx = c_idx

            if best_c_idx >= 0 and (best_combined_sim >= cluster_threshold or len(clusters) >= 6):
                target_c = clusters[best_c_idx]
                cnt = target_c["count"]
                new_centroid = (target_c["centroid"] * cnt + vec) / float(cnt + 1)
                c_norm = float(np.linalg.norm(new_centroid))
                if c_norm > 1e-9:
                    new_centroid = new_centroid / c_norm

                new_disc = (target_c["disc_centroid"] * cnt + disc_vec) / float(cnt + 1)
                d_norm = float(np.linalg.norm(new_disc))
                if d_norm > 1e-9:
                    new_disc = new_disc / d_norm

                target_c["centroid"] = new_centroid
                target_c["disc_centroid"] = new_disc
                target_c["count"] = cnt + 1
                target_c["members"].append(idx)
                segments[idx].speaker_id = target_c["speaker_id"]
                segments[idx].speaker_name = target_c["speaker_name"]
            else:
                c_num = len(clusters) + 1
                c_sid = f"SPEAKER_{next_speaker_slot:02d}"
                next_speaker_slot += 1
                clusters.append(
                    {
                        "speaker_id": c_sid,
                        "speaker_name": f"Người nói {c_num}",
                        "centroid": vec.copy(),
                        "disc_centroid": disc_vec.copy(),
                        "count": 1,
                        "members": [idx],
                    }
                )
                segments[idx].speaker_id = c_sid
                segments[idx].speaker_name = f"Người nói {c_num}"

        valid_unmatched = [i for i in unmatched_indices if seg_disc_vectors[i] is not None]
        if len(clusters) == 1 and len(valid_unmatched) >= 2:
            min_sim = 1.0
            seed_a = valid_unmatched[0]
            seed_b = valid_unmatched[-1]
            for i_pos in range(len(valid_unmatched)):
                for j_pos in range(i_pos + 1, len(valid_unmatched)):
                    ia = valid_unmatched[i_pos]
                    ib = valid_unmatched[j_pos]
                    s_val = cosine_similarity(seg_disc_vectors[ia], seg_disc_vectors[ib])
                    if s_val < min_sim:
                        min_sim = s_val
                        seed_a, seed_b = ia, ib

            if min_sim < 0.985:
                sid_1 = clusters[0]["speaker_id"]
                name_1 = "Người nói 1"
                sid_2 = f"SPEAKER_{next_speaker_slot:02d}"
                name_2 = "Người nói 2"
                next_speaker_slot += 1

                cent_a = seg_disc_vectors[seed_a].copy()
                cent_b = seg_disc_vectors[seed_b].copy()

                for idx in valid_unmatched:
                    dv = seg_disc_vectors[idx]
                    sim_a = cosine_similarity(dv, cent_a)
                    sim_b = cosine_similarity(dv, cent_b)
                    if sim_a >= sim_b:
                        segments[idx].speaker_id = sid_1
                        segments[idx].speaker_name = name_1
                    else:
                        segments[idx].speaker_id = sid_2
                        segments[idx].speaker_name = name_2

    merged_segments: List[Any] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue

        prev = merged_segments[-1]
        same_speaker = (
            prev.speaker_id == seg.speaker_id
            and prev.speaker_name == seg.speaker_name
        )
        prev_st = prev.start_time if prev.start_time is not None else 0.0
        prev_et = prev.end_time if prev.end_time is not None else prev_st
        cur_st = seg.start_time if seg.start_time is not None else prev_et
        cur_et = seg.end_time if seg.end_time is not None else cur_st

        gap = max(0.0, cur_st - prev_et)
        combined_dur = cur_et - prev_st

        if same_speaker and gap <= max_merge_gap and combined_dur <= max_merged_duration:
            prev.end_time = round(cur_et, 2)
            prev.text = f"{prev.text.rstrip()} {seg.text.lstrip()}".strip()
        else:
            merged_segments.append(seg)

    return merged_segments
