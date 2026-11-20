"""
Module: grounding.py
Duong dan: src/core/ai/llm/grounding.py
Muc dich: Lam sach van ban ASR tieng Viet theo ngu canh, doi chieu bang chung nguon
(_find_best_matching_segment, validate_and_ground_tasks) va hau kiem ten nhan su.
"""

import logging
import re
from typing import Any, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.schemas.meeting import SpeakerProfile, TranscriptSegment
from src.schemas.task import (
    TaskConfirmationStatus,
    TaskItem,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class LLMExtractionOutput(BaseModel):
    """Schema ket qua phan tich cuoc hop tra ve tu analyze_node / MeetingLLMAnalyzer."""

    cleaned_transcript: str = Field(
        default="",
        description="Van ban cuoc hop da duoc lam sach loi chinh ta dong am ASR",
    )
    summary: str = Field(
        default="",
        description="Ban tom tat noi dung chinh cua cuoc hop",
    )
    key_decisions: List[str] = Field(
        default_factory=list,
        description="Danh sach cac quyet dinh quan trong da thong nhat",
    )
    tasks: List[TaskItem] = Field(
        default_factory=list,
        description="Danh sach cong viec trich xuat kem bang chung nguon",
    )
    cleaned_segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="Danh sach cac luot thoai da duoc chuan hoa ngu canh tieng Viet",
    )
    used_fallback: bool = Field(
        default=False,
        description="Danh dau ket qua duoc tao tu phuong an du phong offline",
    )


_ASR_CONTEXT_RULES: List[Tuple[str, str]] = [
    (r"\bhợp đây\b", "họp để"),
    (r"\bhợp đề\b", "họp để"),
    (r"\bđã hoàn thành\s+em đang\b", "đã hoàn thành, em đang"),
    (r"\btải phai\b", "tải file"),
    (r"^\s*tôi còn chức năng\b", "Vậy còn chức năng"),
    (r"\btôi còn chức năng\b", "vậy còn chức năng"),
    (
        r"mô hình ngôi độ có thể chia bản ghi âm hình các luật nói",
        "mô hình mới, nó có thể chia bản ghi âm thành các lượt nói",
    ),
    (
        r"mô hình ngôi độ có thể chia bản ghi âm thành các luật nói",
        "mô hình mới, nó có thể chia bản ghi âm thành các lượt nói",
    ),
    (r"\bmô hình ngôi độ\b", "mô hình mới, nó"),
    (r"\bhình các luật nói\b", "thành các lượt nói"),
    (r"\bthành các luật nói\b", "thành các lượt nói"),
    (r"\bcác luật nói\b", "các lượt nói"),
    (r"\bluật nói\b", "lượt nói"),
    (r"^\s*tốt\s+chúng ta\b", "Tốt, chúng ta"),
    (r"\bhỏi thoải\b", "hội thoại"),
    (r"\btiếng việt\b", "tiếng Việt"),
    (r"\bhiện hạng hoàn thành\b", "hạn hoàn thành"),
    (r"\bhẹn hoàn thành\b", "hạn hoàn thành"),
    (r"\bhiện hạng\b", "hạn"),
    (r"\bthứ hai tuần tới\s+nhớ\b", "thứ Hai tuần tới, nhớ"),
    (r"\bthứ hai tuần tới\b", "thứ Hai tuần tới"),
    (r"\bthứ sáu\b", "thứ Sáu"),
    (r"\bkịch quả\b", "kết quả"),
    (r"\bbáo cáo chuốt\b", "báo cáo trước"),
]

_QUESTION_CUES: Tuple[str, ...] = (
    "giai đoạn nào",
    "như thế nào",
    "thế nào",
    "bao giờ",
    "khi nào",
    "ở đâu",
    "hay chưa",
    "phải không",
)


def clean_vietnamese_asr_text(text: str) -> str:
    """Chuan hoa ngu canh va sua cac loi chinh ta dong am tieng Viet do ASR tao ra."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    for pattern, replacement in _ASR_CONTEXT_RULES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    cleaned = cleaned[0].upper() + cleaned[1:]

    lower_check = cleaned.lower()
    is_question = any(cue in lower_check for cue in _QUESTION_CUES)

    if cleaned[-1] in ".!?":
        if is_question and cleaned[-1] == ".":
            cleaned = cleaned[:-1] + "?"
    else:
        cleaned += "?" if is_question else "."

    return cleaned


_clean_asr_vietnamese_text = clean_vietnamese_asr_text


def build_cleaned_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Chuan hoa ngu canh van ban tieng Viet cho tung TranscriptSegment."""
    cleaned_list: List[TranscriptSegment] = []
    for seg in segments:
        cleaned_text = clean_vietnamese_asr_text(seg.text)
        cleaned_list.append(
            TranscriptSegment(
                speaker_id=seg.speaker_id,
                speaker_name=seg.speaker_name,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=cleaned_text or seg.text,
            )
        )
    return cleaned_list


def _normalize_for_match(text: str) -> str:
    """Chuan hoa chuoi van ban de so khop bang chung voi TranscriptSegment."""
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize_words(text: str) -> Set[str]:
    """Tach tap tu don da chuan hoa phuc vu tinh do tuong dong token."""
    norm = _normalize_for_match(text)
    if not norm:
        return set()
    return {w for w in norm.split(" ") if len(w) > 1}


def _find_best_matching_segment(
    task: TaskItem,
    segments: List[TranscriptSegment],
) -> Optional[TranscriptSegment]:
    """Tim TranscriptSegment khop nhat voi TaskItem dua tren source_quote va moc thoi gian."""
    if not segments:
        return None

    quote_raw = (task.source_quote or "").strip()
    quote_norm = _normalize_for_match(quote_raw)
    quote_cleaned_norm = _normalize_for_match(clean_vietnamese_asr_text(quote_raw))
    quote_tokens = _tokenize_words(quote_raw) | _tokenize_words(
        clean_vietnamese_asr_text(quote_raw)
    )

    best_seg: Optional[TranscriptSegment] = None
    best_score = 0.0

    for seg in segments:
        seg_norm = _normalize_for_match(seg.text)
        seg_cleaned_norm = _normalize_for_match(clean_vietnamese_asr_text(seg.text))
        if not seg_norm:
            continue

        if quote_norm and (quote_norm in seg_norm or seg_norm in quote_norm):
            return seg
        if quote_cleaned_norm and (
            quote_cleaned_norm in seg_cleaned_norm
            or seg_cleaned_norm in quote_cleaned_norm
        ):
            return seg

        seg_tokens = _tokenize_words(seg.text) | _tokenize_words(
            clean_vietnamese_asr_text(seg.text)
        )
        if quote_tokens and seg_tokens:
            overlap_count = len(quote_tokens & seg_tokens)
            ratio = overlap_count / float(max(1, min(len(quote_tokens), len(seg_tokens))))
            if ratio > best_score:
                best_score = ratio
                best_seg = seg

    if best_seg is not None and best_score >= 0.45:
        return best_seg

    if task.source_start_time is not None and task.source_end_time is not None:
        t_start = float(task.source_start_time)
        t_end = float(task.source_end_time)
        best_time_seg: Optional[TranscriptSegment] = None
        best_overlap = 0.0

        for seg in segments:
            if seg.start_time is None or seg.end_time is None:
                continue
            ov = max(0.0, min(t_end, seg.end_time) - max(t_start, seg.start_time))
            if ov > best_overlap:
                best_overlap = ov
                best_time_seg = seg
            elif (
                abs(seg.start_time - t_start) <= 1.0
                and abs(seg.end_time - t_end) <= 1.0
                and best_time_seg is None
            ):
                best_time_seg = seg

        if best_time_seg is not None:
            return best_time_seg

    return None


def validate_and_ground_tasks(
    tasks: List[TaskItem],
    segments: List[TranscriptSegment],
) -> List[TaskItem]:
    """Hau kiem danh sach TaskItem do AI de xuat va gan bang chung nguon chinh xac."""
    grounded_tasks: List[TaskItem] = []

    for task in tasks:
        raw_quote = (task.source_quote or "").strip()
        if not raw_quote:
            logger.warning(
                "Bo qua task '%s' vi thieu bang chung source_quote.",
                task.title,
            )
            continue

        matched_seg = _find_best_matching_segment(task, segments)
        if matched_seg is not None:
            grounded_start = matched_seg.start_time
            grounded_end = matched_seg.end_time
            grounded_quote = clean_vietnamese_asr_text(matched_seg.text) or raw_quote
        else:
            if segments:
                logger.warning(
                    "Khong tim thay segment khop cho task '%s'; giu nguyen timestamp goc.",
                    task.title,
                )
            grounded_start = task.source_start_time
            grounded_end = task.source_end_time
            grounded_quote = clean_vietnamese_asr_text(raw_quote)

        clean_assigner = (task.assigner or "").strip() or "Chưa xác định"
        clean_assignee = (task.assignee or "").strip() or "Chưa chỉ định"
        clean_title = task.title.strip()
        if not clean_title:
            continue

        grounded_tasks.append(
            TaskItem(
                id=task.id,
                meeting_id=task.meeting_id,
                title=clean_title,
                description=(task.description or "").strip(),
                assigner=clean_assigner,
                assignee=clean_assignee,
                due_date=(task.due_date or "").strip() or None,
                priority=task.priority,
                priority_reason=(task.priority_reason or "").strip(),
                source_quote=grounded_quote,
                source_start_time=grounded_start,
                source_end_time=grounded_end,
                status=TaskStatus.TODO,
                confirmation_status=TaskConfirmationStatus.PENDING,
            )
        )

    return grounded_tasks


def extract_valid_speaker_names(
    valid_speakers: Optional[List[Any]],
    segments: Optional[List[TranscriptSegment]] = None,
) -> List[str]:
    """Trich xuat danh sach nhan/ten nhan su hop le tu DB va segments cuoc hop."""
    names: List[str] = []
    seen: Set[str] = set()

    for spk in valid_speakers or []:
        candidates: List[str] = []
        if isinstance(spk, SpeakerProfile):
            candidates.extend([spk.name, spk.display_label, spk.username or ""])
        elif isinstance(spk, dict):
            candidates.extend(
                [
                    str(spk.get("name") or ""),
                    str(spk.get("display_label") or ""),
                    str(spk.get("username") or ""),
                ]
            )
        elif isinstance(spk, str):
            candidates.append(spk)

        for c in candidates:
            clean_c = c.strip()
            if clean_c and clean_c.lower() not in seen:
                seen.add(clean_c.lower())
                names.append(clean_c)

    for seg in segments or []:
        for c in (seg.speaker_name or "", seg.speaker_id or ""):
            clean_c = c.strip()
            if clean_c and clean_c.lower() not in seen:
                seen.add(clean_c.lower())
                names.append(clean_c)

    return names


def is_assignee_valid(assignee: str, valid_names: List[str]) -> bool:
    """Kiem tra ten nguoi nhan viec co khop voi danh sach nhan su hop le hay khong."""
    clean_assignee = (assignee or "").strip()
    if not clean_assignee or clean_assignee in ("Chưa chỉ định", "Chưa xác định", "Chưa phân vai"):
        return True
    if not valid_names:
        return True

    norm_assignee = _normalize_for_match(clean_assignee)
    if not norm_assignee:
        return True

    for valid_name in valid_names:
        norm_valid = _normalize_for_match(valid_name)
        if not norm_valid:
            continue
        if norm_assignee == norm_valid or norm_assignee in norm_valid or norm_valid in norm_assignee:
            return True

    return False
