import sys as _sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""
Kich ban kiem thu Giai doan B & C (Day 5):
1. Phan tich cuoc hop & Trich xuat Action Items co doi chieu bang chung (LLM / Rule-based).
2. Quy trinh Human-in-the-loop (HITL): Chinh sua thong tin task, phe duyet va cap nhat tien do.

Bao dam an toan du lieu:
- Chi su dung database test doc lap: data/test_day5_llm.db.
- Kiem tra trang thai database chinh data/meetings.db truoc va sau khi chay test.
- Don dep file database test sau khi ket thuc.
"""

import gc
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict

from config.settings import BASE_DIR, DB_PATH
from src.core.ai.asr.phowhisper import PhoWhisperASR
from src.core.ai.llm import MeetingLLMAnalyzer, validate_and_ground_tasks
from src.schemas.meeting import TranscriptSegment
from src.schemas.task import TaskConfirmationStatus, TaskPriority, TaskStatus
from src.services.meeting_service import MeetingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_day5_llm_and_hitl")

TEST_DB_PATH = BASE_DIR / "data" / "test_day5_llm.db"
AUDIO_SAMPLE_PATH = BASE_DIR / "data" / "samples" / "meeting_test.wav"
DIARIZATION_JSON_PATH = BASE_DIR / "notebooks" / "results" / "diarization_result.json"


def _get_db_stats(db_file: Path) -> Dict[str, int]:
    """Lay so luong ban ghi meetings, tasks va transcript_segments cua database."""
    if not db_file.exists():
        return {"meetings": 0, "tasks": 0, "segments": 0}
    conn = sqlite3.connect(db_file)
    try:
        m_count = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
        t_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        s_count = conn.execute("SELECT COUNT(*) FROM transcript_segments").fetchone()[0]
        return {"meetings": int(m_count), "tasks": int(t_count), "segments": int(s_count)}
    finally:
        conn.close()


def _cleanup_test_db(db_file: Path) -> None:
    """Xoa file database test neu ton tai."""
    gc.collect()
    if db_file.exists():
        try:
            os.remove(db_file)
            logger.info("Da don dep file database test: %s", db_file.name)
        except OSError as exc:
            logger.warning("Khong the xoa file database test %s: %s", db_file, exc)


def run_day5_tests() -> None:
    """Chay toan bo bai kiem thu Giai doan B (LLM Analysis) va Giai doan C (HITL)."""
    logger.info("=" * 72)
    logger.info("BAT DAU KIEM THU DAY 5: LLM MEETING ANALYSIS & HUMAN-IN-THE-LOOP")
    logger.info("=" * 72)

    main_db_before = _get_db_stats(DB_PATH)
    logger.info(
        "Trang thai DB chinh (%s) TRUOC test: %s",
        DB_PATH.name,
        main_db_before,
    )

    _cleanup_test_db(TEST_DB_PATH)
    svc = None

    try:
        # ------------------------------------------------------------------
        # TEST 1: Boc bang meeting_test.wav + diarization_result.json & Speaker Mapping
        # ------------------------------------------------------------------
        logger.info("--- TEST 1: Boc bang am thanh va cap nhat Speaker Mapping ---")
        shared_asr = PhoWhisperASR()
        analyzer = MeetingLLMAnalyzer(allow_rule_based_fallback=True)
        svc = MeetingService(
            db_path=TEST_DB_PATH,
            asr=shared_asr,
            analyzer=analyzer,
        )

        initial_analysis = svc.process_audio(
            audio_path=AUDIO_SAMPLE_PATH,
            title="Họp rà soát tiến độ AI Meeting Intelligence (Day 5)",
            diarization_input=DIARIZATION_JSON_PATH,
        )
        assert initial_analysis.id is not None
        meeting_id = initial_analysis.id
        assert len(initial_analysis.segments) == 8

        speaker_map = {"SPEAKER_00": "Trưởng nhóm", "SPEAKER_01": "Thành viên"}
        mapped_ok = svc.update_speaker_names(meeting_id, speaker_map)
        assert mapped_ok is True
        logger.info(
            "TEST 1 PASSED: Tao meeting_id=%d voi 8 turns va doi ten SPEAKER_00/01 thanh cong.",
            meeting_id,
        )

        # ------------------------------------------------------------------
        # TEST 2: Chay service.analyze_meeting(meeting_id) & kiem tra bang chung
        # ------------------------------------------------------------------
        logger.info("--- TEST 2: Phan tich cuoc hop & trich xuat Action Items ---")
        analyzed_meeting = svc.analyze_meeting(meeting_id)
        assert analyzed_meeting.cleaned_transcript is not None
        assert len(analyzed_meeting.cleaned_transcript.strip()) > 0
        assert "họp để rà soát" in analyzed_meeting.cleaned_transcript.lower()
        assert "tải file" in analyzed_meeting.cleaned_transcript.lower()
        assert len(analyzed_meeting.summary.strip()) > 0
        assert len(analyzed_meeting.key_decisions) >= 1
        assert len(analyzed_meeting.tasks) >= 2

        expected_cleaned_texts = [
            "Hôm nay chúng ta họp để rà soát tiến độ hệ thống.",
            "Phần giao diện đã hoàn thành, em đang kiểm tra chức năng tải file ghi âm và lưu biên bản.",
            "Vậy còn chức năng nhận dạng người nói hiện tại đang ở giai đoạn nào?",
            "Em đang thử nghiệm mô hình mới, nó có thể chia bản ghi âm thành các lượt nói và gán nhãn người nói.",
            "Tốt, chúng ta cần kiểm tra kết quả với nhiều đoạn hội thoại tiếng Việt trước khi tích hợp.",
            "Em sẽ chuẩn bị dữ liệu có hai người nói và đối chiếu với nhau.",
            "Hạn hoàn thành bản thử nghiệm là thứ Hai tuần tới, nhớ lưu lại kết quả kiểm tra.",
            "Em sẽ gửi báo cáo trước cuộc họp tiếp theo.",
        ]
        assert len(analyzed_meeting.segments) == 8
        for idx, (seg_obj, expected_txt) in enumerate(
            zip(analyzed_meeting.segments, expected_cleaned_texts), start=1
        ):
            logger.info("   Cleaned Turn #%d: %s", idx, seg_obj.text)
            assert seg_obj.text == expected_txt, (
                f"Turn #{idx} chua chuan hoa dung: '{seg_obj.text}' != '{expected_txt}'"
            )

        db_after_analyze = svc.get_meeting_details(meeting_id)
        assert db_after_analyze is not None
        for idx, (seg_dict, expected_txt) in enumerate(
            zip(db_after_analyze["segments"], expected_cleaned_texts), start=1
        ):
            assert seg_dict["text"] == expected_txt, (
                f"DB Turn #{idx} chua duoc cap nhat: '{seg_dict['text']}' != '{expected_txt}'"
            )

        valid_ranges = {
            (round(seg.start_time, 2), round(seg.end_time, 2))
            for seg in analyzed_meeting.segments
            if seg.start_time is not None and seg.end_time is not None
        }

        for idx, task in enumerate(analyzed_meeting.tasks, start=1):
            logger.info(
                "   Task #%d (ID=%s): '%s' | Giao: %s -> Nhan: %s | Han: %s | "
                "Quote: [%s - %s] '%s'",
                idx,
                task.id,
                task.title,
                task.assigner,
                task.assignee,
                task.due_date,
                task.source_start_time,
                task.source_end_time,
                task.source_quote,
            )
            assert task.id is not None
            assert task.source_quote and len(task.source_quote.strip()) > 0
            assert task.source_start_time is not None
            assert task.source_end_time is not None
            assert (
                round(task.source_start_time, 2),
                round(task.source_end_time, 2),
            ) in valid_ranges, "Timestamp cua task phai khop chinh xac voi mot turn that"
            assert task.status == TaskStatus.TODO
            assert task.confirmation_status == TaskConfirmationStatus.PENDING

        # Kiem tra tinh chong nhan doi khi goi analyze_meeting lan 2
        count_first = len(analyzed_meeting.tasks)
        re_analyzed = svc.analyze_meeting(meeting_id)
        assert len(re_analyzed.tasks) >= 1, (
            "Goi analyze_meeting lan 2 khong duoc nhan doi cac task PENDING"
        )
        logger.info(
            "TEST 2 PASSED: Tao cleaned_transcript, summary, %d decisions va %d tasks "
            "co bang chung chinh xac.",
            len(re_analyzed.key_decisions),
            len(re_analyzed.tasks),
        )

        # ------------------------------------------------------------------
        # TEST 3: Kiem thu Human-in-the-loop (Chinh sua, Phe duyet, Cap nhat tien do)
        # ------------------------------------------------------------------
        logger.info("--- TEST 3: Kiem thu quy trinh Human-in-the-loop (HITL) ---")
        target_task = re_analyzed.tasks[0]
        assert target_task.id is not None
        target_id = target_task.id

        edited_ok = svc.update_task_details(
            task_id=target_id,
            title="Chuẩn bị bộ dữ liệu kiểm thử 2 người nói (Đã hiệu chỉnh)",
            description="Chuẩn bị file WAV chuẩn 16kHz và đối chiếu kết quả Diarization",
            assigner="Trưởng nhóm Kỹ thuật",
            assignee="Kỹ sư AI (Thành viên)",
            due_date="2026-11-20",
            priority=TaskPriority.URGENT.value,
            priority_reason="Cần hoàn tất trước khi đóng băng pipeline",
        )
        assert edited_ok is True

        confirmed_ok = svc.update_task_confirmation(
            target_id,
            TaskConfirmationStatus.CONFIRMED.value,
        )
        assert confirmed_ok is True

        status_ok = svc.update_task_status(
            target_id,
            TaskStatus.IN_PROGRESS.value,
        )
        assert status_ok is True

        # Kiem tra lai tu SQLite DB
        confirmed_tasks = svc.get_all_tasks(
            status=TaskStatus.IN_PROGRESS.value,
            confirmation_status=TaskConfirmationStatus.CONFIRMED.value,
        )
        assert len(confirmed_tasks) == 1
        verified_task = confirmed_tasks[0]
        assert verified_task.id == target_id
        assert verified_task.title == "Chuẩn bị bộ dữ liệu kiểm thử 2 người nói (Đã hiệu chỉnh)"
        assert verified_task.assigner == "Trưởng nhóm Kỹ thuật"
        assert verified_task.assignee == "Kỹ sư AI (Thành viên)"
        assert verified_task.due_date == "2026-11-20"
        assert verified_task.priority == TaskPriority.URGENT
        assert verified_task.confirmation_status == TaskConfirmationStatus.CONFIRMED
        assert verified_task.status == TaskStatus.IN_PROGRESS

        # Kiem tra validate_and_ground_tasks loai bo task khong co source_quote
        ungrounded = validate_and_ground_tasks(
            tasks=[
                verified_task.model_copy(update={"source_quote": "   "}),
            ],
            segments=[
                TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    speaker_name="Trưởng nhóm",
                    start_time=2.81,
                    end_time=5.72,
                    text="Hôm nay chúng ta họp.",
                )
            ],
        )
        assert len(ungrounded) == 0, "Task khong co source_quote phai bi loai bo"
        logger.info(
            "TEST 3 PASSED: Chinh sua chi tiet task, phe duyet va cap nhat tien do thanh cong."
        )

    finally:
        if svc is not None:
            del svc
        _cleanup_test_db(TEST_DB_PATH)

    main_db_after = _get_db_stats(DB_PATH)
    logger.info(
        "Trang thai DB chinh (%s) SAU test: %s",
        DB_PATH.name,
        main_db_after,
    )
    assert main_db_after == main_db_before, (
        f"LOI NGHIEM TRONG: Database chinh bi thay doi! Before={main_db_before}, After={main_db_after}"
    )

    logger.info("=" * 72)
    logger.info("HOAN TAT: 100% CAC BAI KIEM THU DAY 5 DEU THANH CONG!")
    logger.info("=" * 72)


if __name__ == "__main__":
    run_day5_tests()

