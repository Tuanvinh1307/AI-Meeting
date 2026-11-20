"""
Module: gemini_client.py
Duong dan: src/core/ai/llm/gemini_client.py
Muc dich: Goi REST API Gemini / Groq, chuan hoa ten model va API key,
va phan tich phan hoi JSON thanh LLMExtractionOutput.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from dotenv import load_dotenv

from config.settings import GEMINI_MODEL
from src.core.ai.llm.fallback import rule_based_analyze
from src.core.ai.llm.grounding import (
    LLMExtractionOutput,
    build_cleaned_segments,
    clean_vietnamese_asr_text,
    extract_valid_speaker_names,
    validate_and_ground_tasks,
)
from src.core.ai.llm.prompts import (
    build_formatted_turns,
    build_system_prompt,
    build_user_prompt_with_feedback,
)
from src.schemas.meeting import TranscriptSegment
from src.schemas.task import (
    TaskConfirmationStatus,
    TaskItem,
    TaskPriority,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class MeetingLLMAnalyzer:
    """
    Bo phan tich cuoc hop su dung LLM (Gemini / Groq REST API truc tiep) kem co che
    tu sua loi theo validation_errors trong LangGraph va fallback quy tac ngu nghia tieng Viet.
    """

    GEMINI_MODEL_CANDIDATES = (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    )

    THINKING_MODELS_LOW_LEVEL = (
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    )

    DEPRECATED_LOCKED_MODELS = (
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    )

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        allow_rule_based_fallback: bool = True,
        timeout_s: int = 60,
    ) -> None:
        load_dotenv()
        raw_gemini_key = (
            gemini_api_key
            if gemini_api_key is not None
            else os.getenv("GEMINI_API_KEY", "")
        )
        self.gemini_api_key = raw_gemini_key.strip().strip("\"'").strip()
        raw_groq_key = (
            groq_api_key
            if groq_api_key is not None
            else os.getenv("GROQ_API_KEY", "")
        )
        self.groq_api_key = raw_groq_key.strip().strip("\"'").strip()
        self.allow_rule_based_fallback = allow_rule_based_fallback
        self.timeout_s = timeout_s

    @staticmethod
    def _clean_model_name(model_name: str) -> str:
        """Chuan hoa ten model Gemini: xoa khoang trang, dau nhay va tien to 'models/' neu co."""
        cleaned = (model_name or "").strip().strip("\"'").strip()
        while cleaned.lower().startswith("models/"):
            cleaned = cleaned[len("models/") :].strip()
        return cleaned

    def _get_gemini_model_candidates(self) -> List[str]:
        """Lay danh sach model Gemini 3.x uu tien (bo qua cac model 2.5/2.0 da bi khoa)."""
        env_model = self._clean_model_name(os.getenv("GEMINI_MODEL", "") or GEMINI_MODEL)
        ordered: List[str] = []
        for candidate in (env_model, *self.GEMINI_MODEL_CANDIDATES):
            clean_c = self._clean_model_name(candidate)
            if (
                clean_c
                and clean_c not in self.DEPRECATED_LOCKED_MODELS
                and clean_c not in ordered
            ):
                ordered.append(clean_c)
        return ordered or list(self.GEMINI_MODEL_CANDIDATES)

    def _build_cleaned_segments(
        self,
        segments: List[TranscriptSegment],
    ) -> List[TranscriptSegment]:
        return build_cleaned_segments(segments)

    def _build_formatted_turns(self, segments: List[TranscriptSegment]) -> str:
        return build_formatted_turns(segments)

    def _build_system_prompt(self) -> str:
        return build_system_prompt()

    @staticmethod
    def _extract_json_from_text(text_content: str) -> Dict[str, Any]:
        """Boc tach an toan doi tuong JSON tu chuoi phan hoi (ke ca khi co markdown ```json)."""
        cleaned = (text_content or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned.strip())

    def _call_gemini_api(self, user_prompt: str) -> Dict[str, Any]:
        """Goi REST API Gemini 3.x theo danh sach model uu tien voi JSON mode."""
        last_error: Optional[Exception] = None
        clean_key = (self.gemini_api_key or "").strip().strip("\"'").strip()

        for raw_model in self._get_gemini_model_candidates():
            clean_model = self._clean_model_name(raw_model)
            generation_config: Dict[str, Any] = {
                "responseMimeType": "application/json",
            }
            if clean_model in self.THINKING_MODELS_LOW_LEVEL:
                generation_config["thinkingConfig"] = {"thinkingLevel": "low"}

            payload = {
                "systemInstruction": {
                    "parts": [{"text": self._build_system_prompt()}]
                },
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                "generationConfig": generation_config,
            }
            encoded_payload = json.dumps(payload).encode("utf-8")

            safe_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
            )
            url = f"{safe_url}?key={clean_key}"
            req = urllib.request.Request(
                url,
                data=encoded_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                parts = resp_data["candidates"][0]["content"]["parts"]
                text_content = next(
                    (
                        p["text"]
                        for p in reversed(parts)
                        if isinstance(p, dict) and "text" in p and not p.get("thought")
                    ),
                    parts[-1]["text"],
                )
                parsed_json = self._extract_json_from_text(text_content)
                logger.info(
                    "Gemini API phan hoi thanh cong (HTTP 200) voi model '%s'.",
                    clean_model,
                )
                return parsed_json
            except urllib.error.HTTPError as http_exc:
                last_error = http_exc
                err_body = ""
                try:
                    err_body = http_exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    err_body = str(http_exc)
                logger.warning(
                    "Loi HTTP %s khi goi Gemini URL '%s?key=***' (model='%s'): %s",
                    http_exc.code,
                    safe_url,
                    clean_model,
                    err_body,
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                ValueError,
            ) as exc:
                last_error = exc
                logger.warning(
                    "Model Gemini '%s' tai URL '%s?key=***' gap loi (%s). Chuyen sang model tiep theo...",
                    clean_model,
                    safe_url,
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

    @staticmethod
    def _normalize_summary_text(raw_summary: Any) -> str:
        """
        Chuan hoa truong summary thanh chuoi str thong nhat:
        - Neu LLM tra ve list cac y -> noi thanh cac dong bat dau bang '- '.
        - Ho tro tach ca chuoi phan tach bang '\n', literal '\\n' lan nhieu y gach dau dong
          bi dinh tren cung 1 dong (ke ca khong co dau cham phia truoc).
        - Giu nguyen cap dau in dam '**...**' va dau xuong dong '\n'.
        """
        if raw_summary is None:
            return ""
        bullet_prefix_re = re.compile(r"^(?:[-•]\s+|\*(?!\*)\s+)")

        if isinstance(raw_summary, list):
            items = [
                bullet_prefix_re.sub("", str(item).strip()).strip()
                for item in raw_summary
                if str(item).strip()
            ]
            return "\n".join(f"- {item}" for item in items if item)

        text = str(raw_summary).replace("\\n", "\n").strip()
        if not text:
            return ""

        # Tach cac y gach dau dong bi dinh tren cung 1 dong (sau dau ket cau hoac truoc '- **' / '- ChuHoa')
        text = re.sub(r"([.!?;])\s+(?=(?:[-•]\s+|\*(?!\*)\s+))", r"\1\n", text)
        text = re.sub(r"(?<=\S)\s+(?=[-•]\s+(?:\*\*|[A-ZÀ-Ỹ]))", "\n", text)

        lines = [line.strip() for line in re.split(r"\r?\n+", text) if line.strip()]
        has_bullet = any(bullet_prefix_re.match(line) for line in lines)
        if len(lines) > 1 or has_bullet:
            normalized_lines: List[str] = []
            for line in lines:
                clean_line = bullet_prefix_re.sub("", line).strip()
                if clean_line:
                    normalized_lines.append(f"- {clean_line}")
            return "\n".join(normalized_lines)

        return text

    def _parse_llm_dict(
        self,
        raw_dict: Dict[str, Any],
        segments: List[TranscriptSegment],
    ) -> LLMExtractionOutput:
        """Chuyen doi dict JSON tu LLM sang LLMExtractionOutput co hau kiem."""
        cleaned_segments = build_cleaned_segments(segments)
        cleaned_transcript = str(raw_dict.get("cleaned_transcript") or "").strip()
        if not cleaned_transcript and cleaned_segments:
            cleaned_transcript = "\n".join(
                f"{s.speaker_name or s.speaker_id}: {s.text}" for s in cleaned_segments
            )
        summary = self._normalize_summary_text(raw_dict.get("summary"))
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
            used_fallback=False,
        )

    def _rule_based_analyze(
        self,
        segments: List[TranscriptSegment],
        raw_transcript: str = "",
    ) -> LLMExtractionOutput:
        return rule_based_analyze(segments=segments, raw_transcript=raw_transcript)

    def analyze(
        self,
        segments: List[TranscriptSegment],
        raw_transcript: str = "",
        validation_errors: Optional[List[str]] = None,
        valid_speakers: Optional[List[Any]] = None,
    ) -> LLMExtractionOutput:
        """
        Phan tich danh sach TranscriptSegment cua cuoc hop:
        - Neu validation_errors khac rong, chen danh sach loi vao prompt de Gemini tu sua.
        - Uu tien goi Gemini API hoac Groq API neu co API key hop le.
        - Neu khong co API key hoac gap loi ket noi mang va allow_rule_based_fallback=True,
          chuyen sang phan tich bang rule_based_analyze.
        """
        cleaned_segments = build_cleaned_segments(segments)
        base_prompt = build_formatted_turns(cleaned_segments)
        if not base_prompt.strip():
            base_prompt = clean_vietnamese_asr_text(raw_transcript)

        valid_names = extract_valid_speaker_names(valid_speakers, cleaned_segments)
        user_prompt = build_user_prompt_with_feedback(
            base_prompt=base_prompt,
            valid_names=valid_names,
            validation_errors=validation_errors,
        )

        if self.gemini_api_key:
            try:
                logger.info(
                    "Dang goi Gemini API theo danh sach model uu tien de phan tich cuoc hop..."
                )
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
            return rule_based_analyze(segments=segments, raw_transcript=raw_transcript)

        raise RuntimeError(
            "Khong the phan tich cuoc hop: Chua cau hinh GEMINI_API_KEY / GROQ_API_KEY "
            "va che do fallback dang tat."
        )
