"""
Module: alignment.py
Duong dan: src/ai/alignment.py
Muc dich: Xu ly ket qua Speaker Diarization (lam sach nhieu, gop luot noi)
va ket hop (alignment) voi PhoWhisper ASR de tao danh sach TranscriptSegment
co nhan nguoi noi va moc thoi gian chinh xac.

Quy tac kien truc:
- Khong import Streamlit hoac bat ky thu vien UI nao.
- Khong in truc tiep ra console, su dung logging chuan cua he thong.
- Khong su dung emoji trong code, docstring, log hoac exception.
- Bao toan trung thuc moc thoi gian (khong tu bia timestamp neu la None).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from src.ai.transcription.phowhisper import PhoWhisperASR
from src.schemas.meeting import TranscriptSegment

logger = logging.getLogger(__name__)

DiarizationInputType = Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]


def _resolve_speaker_name(
    speaker_id: str,
    speaker_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Chuyen doi ma dinh danh speaker_id sang ten hien thi than thien.
    Uu tien su dung speaker_mapping neu duoc cung cap.
    """
    if speaker_mapping and speaker_id in speaker_mapping:
        mapped = str(speaker_mapping[speaker_id]).strip()
        if mapped:
            return mapped

    if not speaker_id or speaker_id == "SPEAKER_UNASSIGNED":
        return "Chưa phân vai"

    if speaker_id.startswith("SPEAKER_"):
        suffix = speaker_id.replace("SPEAKER_", "", 1)
        if suffix.isdigit():
            return f"Người nói {int(suffix) + 1}"

    return speaker_id


def load_diarization_segments(
    diarization_input: DiarizationInputType,
    prefer_exclusive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Nap danh sach cac phan doan diarization tu duong dan file JSON, dict hoac list.

    Tham so:
    - diarization_input: Duong dan file JSON (str/Path), dict chua ket qua diarization,
      hoac list cac dict segment.
    - prefer_exclusive: Uu tien lay truong 'exclusive_diarization' khi input la dict.

    Tra ve:
    - Danh sach cac dict bieu dien segment diarization.
    """
    data: Any = diarization_input

    if isinstance(diarization_input, (str, Path)):
        file_path = Path(diarization_input)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay file ket qua diarization tai: {file_path.resolve()}"
            )
        try:
            with open(file_path, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except Exception as exc:
            raise ValueError(
                f"Khong the doc file JSON diarization '{file_path}': {exc}"
            ) from exc

    if isinstance(data, dict):
        selected_list: Optional[List[Dict[str, Any]]] = None
        if prefer_exclusive and isinstance(data.get("exclusive_diarization"), list):
            selected_list = data["exclusive_diarization"]
        elif isinstance(data.get("regular_diarization"), list):
            selected_list = data["regular_diarization"]
        elif isinstance(data.get("segments"), list):
            selected_list = data["segments"]
        elif isinstance(data.get("exclusive_diarization"), list):
            selected_list = data["exclusive_diarization"]

        if selected_list is None:
            raise ValueError(
                "Dict diarization khong chua key hop le "
                "('exclusive_diarization', 'regular_diarization', hoac 'segments')."
            )
        return [dict(item) for item in selected_list if isinstance(item, dict)]

    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]

    raise ValueError(
        f"Dinh dang diarization_input khong hop le: {type(diarization_input)}."
    )


def clean_and_merge_diarization_turns(
    segments: List[Dict[str, Any]],
    min_duration: float = 0.1,
    max_merge_gap: float = 1.2,
) -> List[Dict[str, Any]]:
    """
    Lam sach cac doan am thanh qua ngan (nhieu) va gop cac luot noi lien tiep
    cua cung mot nguoi noi neu khoang nghi giua chung <= max_merge_gap.

    Tham so:
    - segments: Danh sach cac segment tho tu mo hinh diarization.
    - min_duration: Thoi luong toi thieu (giay) de giu lai mot segment.
    - max_merge_gap: Khoang cach toi da (giay) de gop 2 segment cung speaker_id.

    Tra ve:
    - Danh sach cac luot noi (turns) da duoc lam sach va gop.
    """
    if not segments:
        return []

    filtered: List[Dict[str, Any]] = []
    for seg in segments:
        if bool(seg.get("short_segment_flag", False)):
            continue

        if "start" not in seg or "end" not in seg:
            continue

        start_val = float(seg["start"])
        end_val = float(seg["end"])
        actual_duration = end_val - start_val
        recorded_duration = float(seg.get("duration", actual_duration))

        if actual_duration < min_duration or recorded_duration < min_duration:
            continue

        speaker_id = str(seg.get("speaker_id", "SPEAKER_UNASSIGNED")).strip()
        if not speaker_id:
            speaker_id = "SPEAKER_UNASSIGNED"

        filtered.append(
            {
                "start": start_val,
                "end": end_val,
                "speaker_id": speaker_id,
                "duration": round(actual_duration, 3),
                "short_segment_flag": False,
            }
        )

    if not filtered:
        logger.info(
            "Lam sach diarization: %d raw -> 0 hop le sau khi loc.",
            len(segments),
        )
        return []

    filtered.sort(key=lambda item: (item["start"], item["end"]))

    merged: List[Dict[str, Any]] = [dict(filtered[0])]
    for current in filtered[1:]:
        prev = merged[-1]
        gap = current["start"] - prev["end"]

        if current["speaker_id"] == prev["speaker_id"] and gap <= max_merge_gap:
            new_end = max(prev["end"], current["end"])
            prev["end"] = new_end
            prev["duration"] = round(new_end - prev["start"], 3)
        else:
            merged.append(dict(current))

    logger.info(
        "Xu ly diarization hoan tat: raw=%d, filtered=%d (loai bo %d), merged_turns=%d.",
        len(segments),
        len(filtered),
        len(segments) - len(filtered),
        len(merged),
    )
    return merged


def align_segments_by_overlap(
    asr_segments: List[TranscriptSegment],
    diarization_turns: List[Dict[str, Any]],
    min_overlap_ratio: float = 0.1,
    speaker_mapping: Optional[Dict[str, str]] = None,
) -> List[TranscriptSegment]:
    """
    Gan nhan nguoi noi cho danh sach TranscriptSegment co san dua tren do phu
    thoi gian (temporal overlap) voi cac luot noi diarization_turns.

    Quy tac trung thuc du lieu:
    - Neu segment khong co start_time hoac end_time (gia tri la None) hoac
      end_time <= start_time, gan 'SPEAKER_UNASSIGNED' va giu nguyen timestamp goc.
    - Neu ty le giao nhau (overlap / segment_duration) nho hon min_overlap_ratio,
      gan 'SPEAKER_UNASSIGNED'.
    """
    aligned_results: List[TranscriptSegment] = []

    for seg in asr_segments:
        seg_start = seg.start_time
        seg_end = seg.end_time

        if seg_start is None or seg_end is None or seg_end <= seg_start:
            unassigned_id = "SPEAKER_UNASSIGNED"
            aligned_results.append(
                TranscriptSegment(
                    speaker_id=unassigned_id,
                    speaker_name=_resolve_speaker_name(unassigned_id, speaker_mapping),
                    start_time=seg_start,
                    end_time=seg_end,
                    text=seg.text,
                )
            )
            continue

        seg_duration = seg_end - seg_start
        speaker_overlaps: Dict[str, float] = {}

        for turn in diarization_turns:
            if "start" not in turn or "end" not in turn:
                continue
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
            turn_speaker = str(turn.get("speaker_id", "SPEAKER_UNASSIGNED"))

            overlap = max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))
            if overlap > 0.0:
                speaker_overlaps[turn_speaker] = (
                    speaker_overlaps.get(turn_speaker, 0.0) + overlap
                )

        best_speaker = "SPEAKER_UNASSIGNED"
        best_overlap = 0.0
        for spk_id, total_ov in speaker_overlaps.items():
            if total_ov > best_overlap:
                best_overlap = total_ov
                best_speaker = spk_id

        overlap_ratio = best_overlap / seg_duration if seg_duration > 0 else 0.0
        if best_overlap > 0.0 and overlap_ratio >= min_overlap_ratio:
            assigned_id = best_speaker
        else:
            assigned_id = "SPEAKER_UNASSIGNED"

        aligned_results.append(
            TranscriptSegment(
                speaker_id=assigned_id,
                speaker_name=_resolve_speaker_name(assigned_id, speaker_mapping),
                start_time=seg_start,
                end_time=seg_end,
                text=seg.text,
            )
        )

    return aligned_results


def transcribe_with_diarization(
    asr: PhoWhisperASR,
    audio_path: Union[str, Path],
    diarization_input: DiarizationInputType,
    min_duration: float = 0.1,
    max_merge_gap: float = 1.2,
    padding_s: float = 0.15,
    speaker_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[TranscriptSegment]]:
    """
    Ket hop ket qua Speaker Diarization va PhoWhisperASR theo tung luot noi:
    1. Doc va kiem tra file am thanh bang SoundFile.
    2. Nap, loc nhieu va gop cac luot noi cung speaker_id.
    3. Cat audio theo tung turn (kem padding_s an toan o hai dau) va goi PhoWhisperASR.
    4. Tao danh sach TranscriptSegment voi speaker_id, speaker_name, start_time, end_time.

    Tra ve:
    - Tuple[str, List[TranscriptSegment]]: Toan bo van ban ghep noi va danh sach segments.
    """
    path_obj = Path(audio_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Khong tim thay file am thanh tai: {path_obj.resolve()}"
        )

    try:
        audio_data, sample_rate = sf.read(str(path_obj))
    except Exception as exc:
        raise ValueError(
            f"Khong the doc file am thanh '{path_obj}': {exc}"
        ) from exc

    if len(audio_data) == 0:
        raise ValueError(f"File am thanh '{path_obj}' rong (0 samples).")

    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    total_duration = len(audio_data) / float(sample_rate)

    raw_segments = load_diarization_segments(diarization_input)
    merged_turns = clean_and_merge_diarization_turns(
        raw_segments,
        min_duration=min_duration,
        max_merge_gap=max_merge_gap,
    )

    if not merged_turns:
        logger.warning(
            "Khong co luot noi diarization nao hop le sau khi loc; "
            "chuyen sang boc bang toan bo file voi nhan SPEAKER_UNASSIGNED."
        )
        unassigned_name = _resolve_speaker_name("SPEAKER_UNASSIGNED", speaker_mapping)
        return asr.transcribe(
            audio_data,
            return_timestamps=True,
            speaker_id="SPEAKER_UNASSIGNED",
            speaker_name=unassigned_name,
        )

    aligned_segments: List[TranscriptSegment] = []
    text_parts: List[str] = []

    logger.info(
        "Bat dau boc bang theo %d luot noi diarization cho file '%s'...",
        len(merged_turns),
        path_obj.name,
    )

    for idx, turn in enumerate(merged_turns, start=1):
        turn_start = float(turn["start"])
        turn_end = float(turn["end"])
        speaker_id = str(turn.get("speaker_id", "SPEAKER_UNASSIGNED"))
        speaker_name = _resolve_speaker_name(speaker_id, speaker_mapping)

        pad_start = max(0.0, turn_start - padding_s)
        pad_end = min(total_duration, turn_end + padding_s)

        start_sample = max(0, int(pad_start * sample_rate))
        end_sample = min(len(audio_data), int(pad_end * sample_rate))
        chunk_audio = audio_data[start_sample:end_sample]

        if len(chunk_audio) == 0:
            logger.debug("Bo qua turn %d vi doan audio cat ra rong.", idx)
            continue

        chunk_text, _ = asr.transcribe(
            chunk_audio,
            return_timestamps=False,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
        )

        cleaned_text = chunk_text.strip()
        if not cleaned_text:
            logger.debug(
                "Turn %d [%.2fs - %.2fs] (%s) khong nhan dien duoc van ban.",
                idx,
                turn_start,
                turn_end,
                speaker_id,
            )
            continue

        segment = TranscriptSegment(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            start_time=round(turn_start, 2),
            end_time=round(turn_end, 2),
            text=cleaned_text,
        )
        aligned_segments.append(segment)
        text_parts.append(cleaned_text)

    full_text = " ".join(text_parts).strip()
    logger.info(
        "Hoan tat transcribe_with_diarization: thu duoc %d/%d doan hoi thoai co noi dung.",
        len(aligned_segments),
        len(merged_turns),
    )
    return full_text, aligned_segments
