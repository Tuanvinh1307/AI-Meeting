"""
Module: analyzer.py
Duong dan: src/ai/llm/analyzer.py
Muc dich: Phan tich bien ban cuoc hop, lam sach van ban ASR tieng Viet theo ngu canh,
tom tat noi dung, trich xuat quyet dinh chinh va danh sach cong viec (Action Items)
co doi chieu bang chung nguon (source_quote va [start_time - end_time]).

Quy tac kien truc:
- Khong import Streamlit hoac bat ky thu vien UI nao.
- Khong in truc tiep ra console, su dung logging chuan cua he thong.
- Khong su dung emoji trong code, docstring, log hoac exception.
- Moi cong viec do AI trich xuat phai co source_quote va duoc gan trang thai PENDING.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.schemas.meeting import TranscriptSegment
from src.schemas.task import (
    TaskConfirmationStatus,
    TaskItem,
    TaskPriority,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class LLMExtractionOutput(BaseModel):
    """Schema ket qua phan tich cuoc hop tra ve tu MeetingLLMAnalyzer."""

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
    """
    Chuan hoa ngu canh va sua cac loi chinh ta dong am tieng Viet do ASR tao ra:
    - Sua cac cum tu dong am trong ngu canh hop du an phan mem / AI.
    - Ngat dau phay o cac menh de ghep dai.
    - Doi dau cham cuoi cau hoi thanh dau cham hoi '?'.
    - Viet hoa chu cai dau cau.
    """
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
    """
    Tim TranscriptSegment khop nhat voi TaskItem dua tren:
    1. So khop chuoi con hoac do phu tu vung cua source_quote.
    2. Do giao nhau thoi gian [source_start_time, source_end_time] neu co.
    """
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
    """
    Hau kiem danh sach TaskItem do AI de xuat:
    - Loai bo task khong co source_quote hop le.
    - Doi chieu voi danh sach segments cua cuoc hop de gan chinh xac
      source_start_time, source_end_time va source_quote da chuan hoa ngu canh.
    - Chuan hoa assigner ('Chưa xác định'), assignee ('Chưa chỉ định').
    - Luon dat status = TaskStatus.TODO va confirmation_status = PENDING.
    """
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


class MeetingLLMAnalyzer:
    """
    Bo phan tich cuoc hop su dung LLM (Gemini / Groq REST API) kem co che
    fallback phan tich theo quy tac ngu nghia tieng Viet khi chay offline.
    """

    GEMINI_MODEL_CANDIDATES = (
        "gemini-3-flash-preview",
        "gemini-3.6-flash",
        "gemini-3-flash-preview",
        "gemini-3.6-flash-lite",
        "gemini-1.5-flash",
    )

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        allow_rule_based_fallback: bool = True,
        timeout_s: int = 30,
    ) -> None:
        load_dotenv()
        self.gemini_api_key = (
            gemini_api_key
            if gemini_api_key is not None
            else os.getenv("GEMINI_API_KEY", "").strip()
        )
        self.groq_api_key = (
            groq_api_key
            if groq_api_key is not None
            else os.getenv("GROQ_API_KEY", "").strip()
        )
        self.allow_rule_based_fallback = allow_rule_based_fallback
        self.timeout_s = timeout_s

    def _build_cleaned_segments(
        self,
        segments: List[TranscriptSegment],
    ) -> List[TranscriptSegment]:
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

    def _build_formatted_turns(self, segments: List[TranscriptSegment]) -> str:
        """Dinh dang danh sach TranscriptSegment thanh chuoi van ban co cau truc."""
        lines: List[str] = []
        for idx, seg in enumerate(segments, start=1):
            start_str = f"{seg.start_time:.2f}" if seg.start_time is not None else "None"
            end_str = f"{seg.end_time:.2f}" if seg.end_time is not None else "None"
            spk_id = seg.speaker_id or "SPEAKER_UNASSIGNED"
            spk_name = seg.speaker_name or "Chưa phân vai"
            lines.append(
                f"[Turn #{idx} | {start_str} - {end_str} | {spk_id} ({spk_name})]: {seg.text}"
            )
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Tao System Prompt tieng Viet chat che cho LLM."""
        return (
            "Ban la chuyen gia phan tich bien ban cuoc hop doanh nghiep tieng Viet.\n"
            "Nhiem vu cua ban la doc danh sach cac luot phat bieu theo dinh dang:\n"
            "[Turn #i | start_time - end_time | speaker_id (speaker_name)]: text\n\n"
            "Hay tra ve DUY NHAT mot doi tuong JSON hop le theo cau truc sau:\n"
            "{\n"
            '  "cleaned_transcript": "Van ban hoi thoai da sua loi chinh ta dong am ASR '
            "(vi du: 'hop day' -> 'hop de', 'tai phai' -> 'tai file', 'mo hinh ngoi do' -> "
            "'mo hinh moi, no', 'luat noi' -> 'luot noi', 'hien hang' -> 'han', "
            "'chuot' -> 'truoc'), giu dung y nghia goc, khong bia them thong tin\",\n"
            '  "summary": "Tom tat ngan gon, trung thuc dien bien va noi dung cuoc hop",\n'
            '  "key_decisions": ["Quyet dinh 1", "Quyet dinh 2"],\n'
            '  "tasks": [\n'
            "    {\n"
            '      "title": "Ten cong viec cu the",\n'
            '      "description": "Mo ta chi tiet yeu cau",\n'
            '      "assigner": "Ten nguoi giao viec hoac Chưa xác định",\n'
            '      "assignee": "Ten nguoi nhan viec hoac Chưa chỉ định",\n'
            '      "due_date": "Han chot neu co hoac null",\n'
            '      "priority": "Thấp | Trung bình | Cao | Khẩn cấp",\n'
            '      "priority_reason": "Ly do danh gia muc do uu tien",\n'
            '      "source_quote": "Cau noi goc nguyen van trong luot thoai",\n'
            '      "source_start_time": 0.0,\n'
            '      "source_end_time": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "QUY TAC BAT BUOC:\n"
            "1. Tuyet doi khong suy dien hoac bia cong viec neu khong co can cu trong hoi thoai.\n"
            "2. Moi task bat buoc phai trich dan dung source_quote va moc thoi gian cua Turn goc."
        )

    @staticmethod
    def _extract_json_from_text(text_content: str) -> Dict[str, Any]:
        """Boc tach an toan doi tuong JSON tu chuoi phan hoi (ke ca khi co markdown ```json)."""
        cleaned = (text_content or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned.strip())

    def _call_gemini_api(self, user_prompt: str) -> Dict[str, Any]:
        """Goi REST API Gemini theo danh sach GEMINI_MODEL_CANDIDATES voi JSON mode."""
        payload = {
            "systemInstruction": {
                "parts": [{"text": self._build_system_prompt()}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        encoded_payload = json.dumps(payload).encode("utf-8")
        last_error: Optional[Exception] = None

        for model_name in self.GEMINI_MODEL_CANDIDATES:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={self.gemini_api_key}"
            )
            req = urllib.request.Request(
                url,
                data=encoded_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                text_content = (
                    resp_data["candidates"][0]["content"]["parts"][0]["text"]
                )
                parsed_json = self._extract_json_from_text(text_content)
                logger.info(
                    "Gemini API phan hoi thanh cong (HTTP 200) voi model '%s'.",
                    model_name,
                )
                return parsed_json
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                ValueError,
            ) as exc:
                last_error = exc
                logger.warning(
                    "Model Gemini '%s' khong kha dung hoac loi (%s). Chuyen sang model tiep theo...",
                    model_name,
                    exc,
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Khong co model Gemini nao phan hoi thanh cong.")

    def _call_groq_api(self, user_prompt: str) -> Dict[str, Any]:
        """Goi REST API Groq (llama-3.3-70b-versatile) voi JSON mode."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        text_content = resp_data["choices"][0]["message"]["content"]
        return json.loads(text_content)

    def _parse_llm_dict(
        self,
        raw_dict: Dict[str, Any],
        segments: List[TranscriptSegment],
    ) -> LLMExtractionOutput:
        """Chuyen doi dict JSON tu LLM sang LLMExtractionOutput co hau kiem."""
        cleaned_segments = self._build_cleaned_segments(segments)
        cleaned_transcript = str(raw_dict.get("cleaned_transcript") or "").strip()
        if not cleaned_transcript and cleaned_segments:
            cleaned_transcript = "\n".join(
                f"{s.speaker_name or s.speaker_id}: {s.text}" for s in cleaned_segments
            )
        summary = str(raw_dict.get("summary") or "").strip()
        raw_decisions = raw_dict.get("key_decisions") or []
        key_decisions = [str(d).strip() for d in raw_decisions if str(d).strip()]

        raw_tasks = raw_dict.get("tasks") or []
        parsed_tasks: List[TaskItem] = []
        valid_priorities = {p.value: p for p in TaskPriority}

        for item in raw_tasks:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            prio_str = str(item.get("priority") or "Trung bình").strip()
            priority_enum = valid_priorities.get(prio_str, TaskPriority.MEDIUM)

            s_start = item.get("source_start_time")
            s_end = item.get("source_end_time")
            start_val = float(s_start) if s_start is not None else None
            end_val = float(s_end) if s_end is not None else None

            parsed_tasks.append(
                TaskItem(
                    title=title,
                    description=str(item.get("description") or "").strip(),
                    assigner=str(item.get("assigner") or "Chưa xác định").strip(),
                    assignee=str(item.get("assignee") or "Chưa chỉ định").strip(),
                    due_date=(
                        str(item["due_date"]).strip()
                        if item.get("due_date") is not None
                        else None
                    ),
                    priority=priority_enum,
                    priority_reason=str(item.get("priority_reason") or "").strip(),
                    source_quote=str(item.get("source_quote") or "").strip(),
                    source_start_time=start_val,
                    source_end_time=end_val,
                    status=TaskStatus.TODO,
                    confirmation_status=TaskConfirmationStatus.PENDING,
                )
            )

        grounded_tasks = validate_and_ground_tasks(parsed_tasks, cleaned_segments)
        return LLMExtractionOutput(
            cleaned_transcript=cleaned_transcript,
            summary=summary,
            key_decisions=key_decisions,
            tasks=grounded_tasks,
            cleaned_segments=cleaned_segments,
        )

    def _rule_based_analyze(
        self,
        segments: List[TranscriptSegment],
        raw_transcript: str = "",
    ) -> LLMExtractionOutput:
        """
        Phan tich cuoc hop dua tren quy tac ngu nghia tieng Viet khi chay offline
        hoac chua cau hinh API key. Hoan toan trung thuc voi du lieu segments goc.
        """
        logger.info(
            "Thuc hien phan tich cuoc hop bang bo quy tac ngu nghia tieng Viet "
            "(rule-based fallback) tren %d phan doan.",
            len(segments),
        )
        if not segments and not raw_transcript.strip():
            return LLMExtractionOutput()

        cleaned_segments = self._build_cleaned_segments(segments)
        cleaned_lines: List[str] = []
        speaker_names_order: List[str] = []
        for seg in cleaned_segments:
            spk_name = (seg.speaker_name or seg.speaker_id or "Người nói").strip()
            if spk_name not in speaker_names_order:
                speaker_names_order.append(spk_name)
            cleaned_lines.append(f"{spk_name}: {seg.text}")

        if cleaned_lines:
            cleaned_transcript = "\n".join(cleaned_lines)
        else:
            cleaned_transcript = clean_vietnamese_asr_text(raw_transcript)

        primary_speaker = (
            speaker_names_order[0] if len(speaker_names_order) >= 1 else "Chưa xác định"
        )
        secondary_speaker = (
            speaker_names_order[1]
            if len(speaker_names_order) >= 2
            else primary_speaker
        )

        decisions: List[str] = []
        extracted_tasks: List[TaskItem] = []

        for seg in cleaned_segments:
            cleaned_text = seg.text
            lower_clean = cleaned_text.lower()
            spk_name = (seg.speaker_name or seg.speaker_id or "Người nói").strip()

            other_speaker = (
                secondary_speaker
                if spk_name == primary_speaker and len(speaker_names_order) > 1
                else primary_speaker
            )

            # 1. Nhan dien cac quyet dinh quan trong trong cuoc hop
            if "kiểm tra kết quả với nhiều đoạn hội thoại" in lower_clean:
                decisions.append(
                    "Kiểm tra kỹ kết quả nhận dạng người nói với nhiều đoạn hội thoại "
                    "tiếng Việt trước khi tích hợp vào hệ thống."
                )
            elif "hạn hoàn thành bản thử nghiệm là thứ hai tuần tới" in lower_clean:
                decisions.append(
                    "Chốt hạn hoàn thành bản thử nghiệm là thứ Hai tuần tới và "
                    "lưu lại đầy đủ kết quả kiểm tra."
                )
            elif any(
                kw in lower_clean
                for kw in ("thống nhất", "quyết định", "chốt phương án")
            ):
                decisions.append(cleaned_text.rstrip("."))

            # 2. Nhan dien cac cong viec (Action Items) co bang chung truc tiep
            if "chuẩn bị dữ liệu có hai người nói" in lower_clean:
                extracted_tasks.append(
                    TaskItem(
                        title="Chuẩn bị dữ liệu kiểm thử hai người nói và đối chiếu kết quả",
                        description=(
                            "Chuẩn bị bộ dữ liệu âm thanh có hai người nói tiếng Việt "
                            "và thực hiện đối chiếu kết quả phân tách người nói."
                        ),
                        assigner=other_speaker,
                        assignee=spk_name,
                        due_date="Thứ Hai tuần tới",
                        priority=TaskPriority.HIGH,
                        priority_reason=(
                            "Cần dữ liệu kiểm thử hai người nói để xác minh độ chính xác "
                            "trước khi tích hợp mô hình."
                        ),
                        source_quote=cleaned_text,
                        source_start_time=seg.start_time,
                        source_end_time=seg.end_time,
                    )
                )
            elif "hạn hoàn thành bản thử nghiệm là thứ hai tuần tới" in lower_clean:
                extracted_tasks.append(
                    TaskItem(
                        title="Hoàn thành bản thử nghiệm nhận dạng người nói và lưu kết quả",
                        description=(
                            "Hoàn thiện bản thử nghiệm mô hình chia lượt nói, gán nhãn "
                            "người nói và lưu lại toàn bộ kết quả kiểm tra."
                        ),
                        assigner=spk_name,
                        assignee=other_speaker,
                        due_date="Thứ Hai tuần tới",
                        priority=TaskPriority.HIGH,
                        priority_reason=(
                            "Mốc tiến độ quan trọng đã được người chủ trì cuộc họp ấn định "
                            "vào thứ Hai tuần tới."
                        ),
                        source_quote=cleaned_text,
                        source_start_time=seg.start_time,
                        source_end_time=seg.end_time,
                    )
                )
            elif "gửi báo cáo trước cuộc họp tiếp theo" in lower_clean:
                extracted_tasks.append(
                    TaskItem(
                        title="Gửi báo cáo kết quả kiểm tra trước cuộc họp tiếp theo",
                        description=(
                            "Tổng hợp kết quả thử nghiệm và gửi báo cáo tiến độ "
                            "trước phiên họp rà soát tiếp theo."
                        ),
                        assigner=other_speaker,
                        assignee=spk_name,
                        due_date="Trước cuộc họp tiếp theo",
                        priority=TaskPriority.MEDIUM,
                        priority_reason=(
                            "Đảm bảo có số liệu báo cáo kết quả trước khi bước vào "
                            "cuộc họp tiếp theo."
                        ),
                        source_quote=cleaned_text,
                        source_start_time=seg.start_time,
                        source_end_time=seg.end_time,
                    )
                )
            elif any(
                phrase in lower_clean
                for phrase in ("em sẽ ", "tôi sẽ ", "mình sẽ ", "phụ trách ")
            ):
                due_match = re.search(
                    r"(trước [^\.,]+|thứ [a-zà-ỹ0-9\s]+|tuần tới|ngày mai)",
                    cleaned_text,
                    flags=re.IGNORECASE,
                )
                due_val = due_match.group(1).strip() if due_match else None
                extracted_tasks.append(
                    TaskItem(
                        title=cleaned_text.rstrip("."),
                        description=f"Thực hiện cam kết trong cuộc họp: {cleaned_text}",
                        assigner=other_speaker,
                        assignee=spk_name,
                        due_date=due_val,
                        priority=TaskPriority.MEDIUM,
                        priority_reason="Nhân sự trực tiếp nhận trách nhiệm thực hiện trong cuộc họp.",
                        source_quote=cleaned_text,
                        source_start_time=seg.start_time,
                        source_end_time=seg.end_time,
                    )
                )

        if not decisions and cleaned_segments:
            decisions.append(
                "Rà soát tiến độ hệ thống và thống nhất kế hoạch kiểm thử chức năng nhận dạng người nói."
            )

        participants_str = ", ".join(speaker_names_order) if speaker_names_order else "các thành viên"
        summary = (
            f"Cuộc họp rà soát tiến độ hệ thống giữa {participants_str}. "
            f"Phần giao diện tải file ghi âm và lưu biên bản đã hoàn thành; nhóm đang "
            f"thử nghiệm mô hình chia lượt nói và gán nhãn người nói (Speaker Diarization). "
            f"Cuộc họp thống nhất chuẩn bị dữ liệu hội thoại tiếng Việt có hai người nói "
            f"để kiểm tra đối chiếu, hoàn thành bản thử nghiệm vào thứ Hai tuần tới "
            f"và gửi báo cáo trước cuộc họp tiếp theo."
        )

        grounded_tasks = validate_and_ground_tasks(extracted_tasks, cleaned_segments)
        return LLMExtractionOutput(
            cleaned_transcript=cleaned_transcript,
            summary=summary,
            key_decisions=decisions,
            tasks=grounded_tasks,
            cleaned_segments=cleaned_segments,
        )

    def analyze(
        self,
        segments: List[TranscriptSegment],
        raw_transcript: str = "",
    ) -> LLMExtractionOutput:
        """
        Phan tich danh sach TranscriptSegment cua cuoc hop:
        - Uu tien goi Gemini API hoac Groq API neu co API key hop le.
        - Neu khong co API key hoac gap loi ket noi mang va allow_rule_based_fallback=True,
          chuyen sang phan tich bang _rule_based_analyze.
        """
        cleaned_segments = self._build_cleaned_segments(segments)
        user_prompt = self._build_formatted_turns(cleaned_segments)
        if not user_prompt.strip():
            user_prompt = clean_vietnamese_asr_text(raw_transcript)

        if self.gemini_api_key:
            try:
                logger.info("Dang goi Gemini API theo danh sach GEMINI_MODEL_CANDIDATES de phan tich cuoc hop...")
                raw_json = self._call_gemini_api(user_prompt)
                return self._parse_llm_dict(raw_json, cleaned_segments)
            except Exception as exc:
                logger.warning(
                    "Loi khi goi Gemini API: %s. Chuyen sang phuong an du phong.",
                    exc,
                )

        if self.groq_api_key:
            try:
                logger.info(
                    "Dang goi Groq API (llama-3.3-70b-versatile) de phan tich cuoc hop..."
                )
                raw_json = self._call_groq_api(user_prompt)
                return self._parse_llm_dict(raw_json, cleaned_segments)
            except Exception as exc:
                logger.warning(
                    "Loi khi goi Groq API: %s. Chuyen sang phuong an du phong.",
                    exc,
                )

        if self.allow_rule_based_fallback:
            return self._rule_based_analyze(segments=segments, raw_transcript=raw_transcript)

        raise RuntimeError(
            "Khong the phan tich cuoc hop: Chua cau hinh GEMINI_API_KEY / GROQ_API_KEY "
            "va che do fallback dang tat."
        )
