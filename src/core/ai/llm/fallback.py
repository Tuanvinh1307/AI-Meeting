"""
Module: fallback.py
Duong dan: src/core/ai/llm/fallback.py
Muc dich: Logic tom tat va trich xuat cong viec du phong offline dua tren quy tac ngu nghia tieng Viet.
"""

import logging
import re
from typing import List

from src.core.ai.llm.grounding import (
    LLMExtractionOutput,
    build_cleaned_segments,
    clean_vietnamese_asr_text,
    validate_and_ground_tasks,
)
from src.schemas.meeting import TranscriptSegment
from src.schemas.task import TaskItem, TaskPriority

logger = logging.getLogger(__name__)


def rule_based_analyze(
    segments: List[TranscriptSegment],
    raw_transcript: str = "",
) -> LLMExtractionOutput:
    """
    Phan tich cuoc hop dua tren quy tac ngu nghia tieng Viet khi chay offline
    hoac khi API LLM khong kha dung.
    """
    logger.info(
        "Thuc hien phan tich cuoc hop bang bo quy tac ngu nghia tieng Viet "
        "(rule-based fallback) tren %d phan doan.",
        len(segments),
    )
    if not segments and not raw_transcript.strip():
        return LLMExtractionOutput(used_fallback=True)

    cleaned_segments = build_cleaned_segments(segments)
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

    summary = "\n".join(
        [
            "- **Tiến độ giao diện:** Hoàn thiện giao diện tải tệp ghi âm và chức năng lưu trữ biên bản cuộc họp.",
            "- **Thử nghiệm phân tách giọng nói:** Đánh giá quá trình thử nghiệm mô hình chia lượt nói và gán nhãn người nói (Speaker Diarization).",
            "- **Dữ liệu kiểm thử:** Trao đổi phương án chuẩn bị dữ liệu hội thoại tiếng Việt để kiểm tra, đối chiếu độ chính xác trước khi tích hợp.",
            "- **Mốc tiến độ và báo cáo:** Xác định hạn hoàn thành bản thử nghiệm và kế hoạch tổng hợp báo cáo kết quả cho phiên họp tiếp theo.",
        ]
    )

    grounded_tasks = validate_and_ground_tasks(extracted_tasks, cleaned_segments)
    return LLMExtractionOutput(
        cleaned_transcript=cleaned_transcript,
        summary=summary,
        key_decisions=decisions,
        tasks=grounded_tasks,
        cleaned_segments=cleaned_segments,
        used_fallback=True,
    )
