"""
Module: nodes.py
Duong dan: src/core/ai/graph/nodes.py
Muc dich: Dinh nghia 4 Node xu ly trong LangGraph Pipeline kem log do hieu nang chi tiet
(diarization, asr, voiceprint, llm, total_sec, audio_duration_sec, RTF):
1. transcribe_node
2. speaker_align_node
3. analyze_node
4. validate_node
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import soundfile as sf

from src.core.ai.asr.phowhisper import PhoWhisperASR
from src.core.ai.diarization.alignment import (
    align_segments_by_overlap,
    clean_and_merge_diarization_turns,
    load_diarization_segments,
    transcribe_with_diarization,
)
from src.core.ai.diarization.pyannote_diarizer import diarize_audio_with_pyannote
from src.core.ai.diarization.voiceprint import match_speakers_to_profiles
from src.core.ai.graph.state import MeetingGraphState
from src.core.ai.llm.gemini_client import MeetingLLMAnalyzer
from src.core.ai.llm.grounding import (
    LLMExtractionOutput,
    extract_valid_speaker_names,
    is_assignee_valid,
)
from src.schemas.meeting import TranscriptSegment

logger = logging.getLogger(__name__)

_default_asr: Optional[PhoWhisperASR] = None
_default_analyzer: Optional[MeetingLLMAnalyzer] = None


def _get_default_asr() -> PhoWhisperASR:
    global _default_asr
    if _default_asr is None:
        _default_asr = PhoWhisperASR()
    return _default_asr


def _get_default_analyzer() -> MeetingLLMAnalyzer:
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = MeetingLLMAnalyzer()
    return _default_analyzer


def _report_progress(state: MeetingGraphState, step: str, progress: int) -> None:
    """Cap nhat tien do theo tung node LangGraph qua callback neu duoc truyen vao state."""
    cb = state.get("progress_callback")
    if callable(cb):
        try:
            cb(step, int(progress))
        except Exception as exc:
            logger.debug("Khong the cap nhat progress_callback (%s, %d): %s", step, progress, exc)


def _get_audio_duration_sec(audio_path: Optional[Path]) -> float:
    """Lay thoi luong thuc te cua file audio (giay)."""
    if audio_path is None or not audio_path.exists():
        return 0.0
    try:
        info = sf.info(str(audio_path))
        return round(float(info.duration), 2)
    except Exception:
        return 0.0


def _log_pipeline_perf_summary(state: MeetingGraphState, timings: Dict[str, float]) -> None:
    """In 1 dong log tong ket hieu nang toan bo pipeline LangGraph (kem RTF)."""
    diar_s = round(float(timings.get("diarization", 0.0)), 2)
    asr_s = round(float(timings.get("asr", 0.0)), 2)
    vp_s = round(float(timings.get("voiceprint", 0.0)), 2)
    llm_s = round(float(timings.get("llm", 0.0)), 2)
    start_ts = timings.get("pipeline_start_ts")
    if start_ts is not None:
        total_sec = round(max(0.001, time.perf_counter() - float(start_ts)), 2)
    else:
        total_sec = round(diar_s + asr_s + vp_s + llm_s, 2)

    audio_dur = round(float(state.get("audio_duration_sec") or 0.0), 2)
    rtf = round(total_sec / audio_dur, 2) if audio_dur > 0 else 0.0

    perf_line = (
        f"[PIPELINE_PERF] meeting_id={state.get('meeting_id') or 'N/A'} | "
        f"audio_duration_sec={audio_dur:.2f}s | diarization={diar_s:.2f}s | "
        f"asr={asr_s:.2f}s | voiceprint={vp_s:.2f}s | llm={llm_s:.2f}s | "
        f"total_sec={total_sec:.2f}s | RTF={rtf:.2f}"
    )
    print(perf_line, flush=True)
    logger.info(perf_line)


def transcribe_node(state: MeetingGraphState) -> Dict[str, Any]:
    """
    Node 1:
    - Neu co du lieu phan chia nguoi noi (hoac pyannote tu dong):
      Gop cac luot noi lien tiep cua cung 1 speaker truoc, cat audio theo dung tung luot noi
      (cong dem 0.25s o dau va cuoi moi luot) va boc bang tung luot bang PhoWhisper (faster-whisper int8).
    - Neu khong co tep phan chia nguoi noi:
      Chay PhoWhisper truc tiep voi Silero VAD va native timestamp.
    """
    pipeline_start_ts = time.perf_counter()
    timings: Dict[str, float] = dict(state.get("timings") or {})
    timings["pipeline_start_ts"] = pipeline_start_ts
    timings.setdefault("diarization", 0.0)
    timings.setdefault("asr", 0.0)
    timings.setdefault("voiceprint", 0.0)
    timings.setdefault("llm", 0.0)

    existing_aligned = state.get("aligned_segments") or []
    existing_raw = state.get("raw_segments") or []
    if existing_aligned or existing_raw:
        segs = existing_raw or existing_aligned
        text = state.get("transcript_text") or "\n".join(s.text for s in segs if s.text)
        return {
            "raw_segments": segs,
            "transcript_text": text,
            "timings": timings,
            "audio_duration_sec": float(state.get("audio_duration_sec") or 0.0),
        }

    audio_path_raw = state.get("audio_path")
    if not audio_path_raw:
        raise ValueError("Thieu audio_path va raw_segments trong MeetingGraphState.")

    path_obj = Path(audio_path_raw)
    if not path_obj.exists():
        raise FileNotFoundError(f"Khong tim thay file am thanh tai: {path_obj.resolve()}")

    audio_duration_sec = _get_audio_duration_sec(path_obj)
    asr = state.get("asr") or _get_default_asr()
    diarization_source = (
        state.get("diarization_input")
        or state.get("diarization_segments")
        or state.get("diarization_turns")
    )
    speaker_mapping = state.get("speaker_mapping")

    # Bước 1A: Diarization (nếu chưa có file phân vai và có HF_TOKEN)
    diar_t0 = time.perf_counter()
    if diarization_source is None:
        hf_token = os.getenv("HF_TOKEN", "").strip().strip("\"'").strip()
        if hf_token and hf_token != "your_huggingface_token_here":
            _report_progress(state, "diarization", 15)
            try:
                pyannote_turns = diarize_audio_with_pyannote(
                    audio_path=path_obj,
                    hf_token=hf_token,
                )
                if pyannote_turns:
                    diarization_source = pyannote_turns
                    logger.info(
                        "Pyannote tu dong phan chia duoc %d luot noi cho file '%s'.",
                        len(pyannote_turns),
                        path_obj.name,
                    )
            except Exception as exc:
                logger.warning(
                    "Khong the chay pyannote/speaker-diarization-3.1 (%s). Fallback ve chay PhoWhisper truc tiep.",
                    exc,
                )
        else:
            logger.warning(
                "Khong tim thay HF_TOKEN hop le trong bien moi truong. Fallback ve chay PhoWhisper truc tiep."
            )
    else:
        _report_progress(state, "diarization", 20)
    diar_sec = round(time.perf_counter() - diar_t0, 2)
    timings["diarization"] = diar_sec
    logger.info("[PERF] Step diarization: %.2fs", diar_sec)

    # Bước 1B: ASR (PhoWhisper faster-whisper int8)
    _report_progress(state, "asr", 35)
    asr_t0 = time.perf_counter()
    if diarization_source is not None:
        full_text, segments = transcribe_with_diarization(
            asr=asr,
            audio_path=path_obj,
            diarization_input=diarization_source,
            padding_s=0.25,
            speaker_mapping=speaker_mapping,
        )
    else:
        full_text, segments = asr.transcribe(
            path_obj,
            return_timestamps=True,
            speaker_id="SPEAKER_UNASSIGNED",
            speaker_name="Chưa phân vai",
        )
    asr_sec = round(time.perf_counter() - asr_t0, 2)
    timings["asr"] = asr_sec

    if audio_duration_sec <= 0.0 and segments:
        max_end = max((float(s.end_time or 0.0) for s in segments), default=0.0)
        if max_end > 0:
            audio_duration_sec = round(max_end, 2)

    logger.info(
        "[PERF] Step asr: %.2fs (audio_duration_sec=%.2fs, segments=%d)",
        asr_sec,
        audio_duration_sec,
        len(segments),
    )

    if not full_text.strip():
        raise ValueError(
            "Mo hinh khong nhan dien duoc bat ky noi dung am thanh nao tu file ghi am."
        )

    _report_progress(state, "asr", 55)
    out_state: Dict[str, Any] = {
        "raw_segments": segments,
        "transcript_text": full_text,
        "timings": timings,
        "audio_duration_sec": audio_duration_sec,
    }
    if diarization_source is not None:
        out_state["diarization_segments"] = diarization_source
    return out_state


def speaker_align_node(state: MeetingGraphState) -> Dict[str, Any]:
    """
    Node 2: Doi chieu voiceprint.py (top 3 doan dai & ro nhat cho moi speaker)
    de map dung ten tung nhan su.
    """
    _report_progress(state, "voiceprint", 65)
    vp_t0 = time.perf_counter()
    timings: Dict[str, float] = dict(state.get("timings") or {})

    already_aligned = state.get("aligned_segments")
    if already_aligned:
        text = state.get("transcript_text") or "\n".join(
            s.text for s in already_aligned if s.text
        )
        vp_sec = round(time.perf_counter() - vp_t0, 2)
        timings["voiceprint"] = vp_sec
        logger.info("[PERF] Step voiceprint: %.2fs", vp_sec)
        _report_progress(state, "voiceprint", 75)
        if state.get("run_llm_analysis") is False:
            _log_pipeline_perf_summary(state, timings)
        return {
            "aligned_segments": already_aligned,
            "transcript_text": text,
            "timings": timings,
        }

    segments = [
        TranscriptSegment(
            speaker_id=s.speaker_id,
            speaker_name=s.speaker_name,
            start_time=s.start_time,
            end_time=s.end_time,
            text=s.text,
        )
        for s in (state.get("raw_segments") or [])
    ]
    speaker_mapping = state.get("speaker_mapping")
    diarization_turns = state.get("diarization_turns")
    diarization_segments = state.get("diarization_segments")

    if not diarization_turns and diarization_segments is not None:
        try:
            raw_diar = load_diarization_segments(diarization_segments)
            diarization_turns = clean_and_merge_diarization_turns(raw_diar)
        except Exception as exc:
            logger.warning("Khong the doc diarization_segments: %s", exc)

    if diarization_turns and segments and all(
        (not s.speaker_id or s.speaker_id == "SPEAKER_UNASSIGNED") for s in segments
    ):
        segments = align_segments_by_overlap(
            asr_segments=segments,
            diarization_turns=diarization_turns,
            speaker_mapping=speaker_mapping,
        )

    audio_path_raw = state.get("audio_path")
    valid_speakers = state.get("valid_speakers") or []

    has_diarization = (
        state.get("diarization_input") is not None
        or state.get("diarization_segments") is not None
        or bool(diarization_turns)
        or any(
            s.speaker_id and s.speaker_id != "SPEAKER_UNASSIGNED"
            for s in segments
        )
    )

    if audio_path_raw and valid_speakers and segments:
        path_obj = Path(audio_path_raw)
        if path_obj.exists():
            try:
                if has_diarization:
                    matched = match_speakers_to_profiles(
                        audio_path=path_obj,
                        segments=segments,
                        profiles=valid_speakers,
                    )
                    if matched:
                        for seg in segments:
                            if speaker_mapping and seg.speaker_id in speaker_mapping:
                                continue
                            if seg.speaker_id in matched:
                                seg.speaker_name = matched[seg.speaker_id]["display_label"]
                        logger.info(
                            "Tu dong nhan dien giong noi thanh cong cho %d nguoi noi: %s",
                            len(matched),
                            {k: v["display_label"] for k, v in matched.items()},
                        )
                else:
                    for idx, seg in enumerate(segments):
                        if seg.start_time is None or seg.end_time is None:
                            continue
                        dur = float(seg.end_time) - float(seg.start_time)
                        if dur < 1.2 or dur > 25.0:
                            continue
                        temp_sid = f"SEG_{idx:03d}"
                        probe_seg = TranscriptSegment(
                            speaker_id=temp_sid,
                            speaker_name=seg.speaker_name,
                            start_time=seg.start_time,
                            end_time=seg.end_time,
                            text=seg.text,
                        )
                        matched_single = match_speakers_to_profiles(
                            audio_path=path_obj,
                            segments=[probe_seg],
                            profiles=valid_speakers,
                            threshold=0.72,
                        )
                        if temp_sid in matched_single:
                            seg.speaker_name = matched_single[temp_sid]["display_label"]
            except Exception as exc:
                logger.warning("Khong the doi sanh voiceprint tu dong: %s", exc)

    transcript_text = "\n".join(s.text for s in segments if s.text) or (
        state.get("transcript_text") or ""
    )
    vp_sec = round(time.perf_counter() - vp_t0, 2)
    timings["voiceprint"] = vp_sec
    logger.info("[PERF] Step voiceprint: %.2fs", vp_sec)

    _report_progress(state, "voiceprint", 75)
    if state.get("run_llm_analysis") is False:
        _log_pipeline_perf_summary(state, timings)

    return {
        "aligned_segments": segments,
        "transcript_text": transcript_text,
        "timings": timings,
    }


def analyze_node(state: MeetingGraphState) -> Dict[str, Any]:
    """
    Node 3: Tom tat cuoc hop va trich xuat cong viec bang Gemini / LLM Analyzer.
    """
    if state.get("run_llm_analysis") is False:
        return {
            "analysis_result": None,
            "validation_errors": [],
        }

    llm_t0 = time.perf_counter()
    timings: Dict[str, float] = dict(state.get("timings") or {})

    analyzer = state.get("analyzer") or _get_default_analyzer()
    aligned_segments = state.get("aligned_segments") or state.get("raw_segments") or []
    transcript_text = state.get("transcript_text") or ""
    prev_errors = state.get("validation_errors") or []
    valid_speakers = state.get("valid_speakers") or []
    current_retry = int(state.get("retry_count") or 0)

    if prev_errors:
        current_retry += 1
        logger.info(
            "LangGraph analyze_node chay lai (retry_count=%d) de sua %d loi validation...",
            current_retry,
            len(prev_errors),
        )

    _report_progress(state, "llm", 82 if current_retry == 0 else min(90, 82 + current_retry * 4))
    extraction = analyzer.analyze(
        segments=aligned_segments,
        raw_transcript=transcript_text,
        validation_errors=prev_errors if prev_errors else None,
        valid_speakers=valid_speakers,
    )

    llm_step_sec = round(time.perf_counter() - llm_t0, 2)
    timings["llm"] = round(float(timings.get("llm", 0.0)) + llm_step_sec, 2)
    logger.info("[PERF] Step llm (attempt=%d): %.2fs", current_retry + 1, llm_step_sec)

    return {
        "analysis_result": extraction,
        "retry_count": current_retry,
        "timings": timings,
    }


def validate_node(state: MeetingGraphState) -> Dict[str, Any]:
    """
    Node 4: Kiem tra ket qua tu analyze_node (schema, summary khong rong, assignee hop le)
    va in dong log tong ket hieu nang (kem RTF) khi ket thuc pipeline.
    """
    if state.get("run_llm_analysis") is False:
        return {"validation_errors": []}

    _report_progress(state, "llm", 92)
    timings: Dict[str, float] = dict(state.get("timings") or {})

    result = state.get("analysis_result")
    aligned_segments = state.get("aligned_segments") or state.get("raw_segments") or []
    valid_speakers = state.get("valid_speakers") or []
    errors: List[str] = []

    if result is None or not isinstance(result, LLMExtractionOutput):
        errors.append("Ket qua phan tich khong dung dinh dang schema LLMExtractionOutput.")
        return {"validation_errors": errors}

    if not result.summary or not result.summary.strip():
        errors.append("Phan tom tat cuoc hop (summary) khong duoc de trong.")

    valid_names = extract_valid_speaker_names(valid_speakers, aligned_segments)
    if valid_speakers and valid_names:
        for task in result.tasks:
            if not is_assignee_valid(task.assignee, valid_names):
                errors.append(
                    f"Nguoi nhan viec '{task.assignee}' cua cong viec '{task.title}' "
                    f"khong khop voi danh sach nhan su hop le ({', '.join(valid_names)})."
                )

    retry_count = int(state.get("retry_count") or 0)
    used_fallback = bool(getattr(result, "used_fallback", False))
    will_retry = bool(errors and not used_fallback and retry_count < 2)
    if not will_retry:
        _log_pipeline_perf_summary(state, timings)

    return {"validation_errors": errors}
