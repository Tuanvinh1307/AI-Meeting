import sys as _sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""
Kich ban kiem thu tich hop Day 4: Speaker Diarization & PhoWhisper ASR Alignment.
Bao dam an toan du lieu:
- Chi su dung database test doc lap: data/test_day4_alignment.db.
- Kiem tra trang thai database chinh data/meetings.db truoc va sau khi chay test.
- Don dep file database test sau khi ket thuc.
"""

import gc
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict

from config.settings import BASE_DIR, DB_PATH
from src.core.ai.asr.phowhisper import PhoWhisperASR
from src.core.ai.diarization.alignment import (
    align_segments_by_overlap,
    clean_and_merge_diarization_turns,
    load_diarization_segments,
)
from src.schemas.meeting import MeetingAnalysis, TranscriptSegment
from src.services.meeting_service import MeetingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_day4_alignment")

TEST_DB_PATH = BASE_DIR / "data" / "test_day4_alignment.db"
AUDIO_SAMPLE_PATH = BASE_DIR / "data" / "samples" / "meeting_test.wav"
DIARIZATION_JSON_PATH = BASE_DIR / "notebooks" / "results" / "diarization_result.json"
ALIGNED_REF_JSON_PATH = BASE_DIR / "notebooks" / "results" / "aligned_transcript.json"


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


def run_day4_tests() -> None:
    """Chay toan bo cac bai kiem thu Day 4 cho module alignment va MeetingService."""
    logger.info("=" * 72)
    logger.info("BAT DAU KIEM THU DAY 4: DIARIZATION & ASR ALIGNMENT PIPELINE")
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
        # TEST 1: Kiem thu load_diarization_segments & clean_and_merge_diarization_turns
        # ------------------------------------------------------------------
        logger.info("--- TEST 1: Lam sach nhieu va gop luot noi (20 raw -> 8 turns) ---")
        raw_segments = load_diarization_segments(DIARIZATION_JSON_PATH, prefer_exclusive=True)
        assert len(raw_segments) == 20, f"Ky vong 20 raw segments, nhan duoc {len(raw_segments)}"

        raw_snapshot = [dict(item) for item in raw_segments]
        merged_turns = clean_and_merge_diarization_turns(
            raw_segments,
            min_duration=0.1,
            max_merge_gap=1.2,
        )
        assert raw_segments == raw_snapshot, "Ham clean_and_merge khong duoc lam thay doi list dau vao"
        assert len(merged_turns) == 8, f"Ky vong 8 merged turns, nhan duoc {len(merged_turns)}"

        with open(ALIGNED_REF_JSON_PATH, "r", encoding="utf-8") as ref_file:
            ref_data = json.load(ref_file)
        ref_segments = ref_data["segments"]

        for idx, (turn, ref_seg) in enumerate(zip(merged_turns, ref_segments)):
            assert turn["speaker_id"] == ref_seg["speaker_id"], (
                f"Turn {idx} sai speaker_id: {turn['speaker_id']} != {ref_seg['speaker_id']}"
            )
            assert round(turn["start"], 2) == ref_seg["start_time"], (
                f"Turn {idx} sai start_time: {round(turn['start'], 2)} != {ref_seg['start_time']}"
            )
            assert round(turn["end"], 2) == ref_seg["end_time"], (
                f"Turn {idx} sai end_time: {round(turn['end'], 2)} != {ref_seg['end_time']}"
            )
        logger.info("TEST 1 PASSED: 20 raw segments -> loc 5 nhieu -> gop chinh xac 8 turns.")

        # ------------------------------------------------------------------
        # TEST 2: Kiem thu align_segments_by_overlap (bao gom None timestamps)
        # ------------------------------------------------------------------
        logger.info("--- TEST 2: Kiem thu align_segments_by_overlap ---")
        mock_asr_segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_UNASSIGNED",
                speaker_name="Chưa phân vai",
                start_time=2.9,
                end_time=5.5,
                text="Doan thoai cua nguoi noi 1.",
            ),
            TranscriptSegment(
                speaker_id="SPEAKER_UNASSIGNED",
                speaker_name="Chưa phân vai",
                start_time=9.0,
                end_time=15.0,
                text="Doan thoai cua nguoi noi 2.",
            ),
            TranscriptSegment(
                speaker_id="SPEAKER_UNASSIGNED",
                speaker_name="Chưa phân vai",
                start_time=17.0,
                end_time=20.0,
                text="Doan thoai nam trong khoang lang khong co diarization.",
            ),
            TranscriptSegment(
                speaker_id="SPEAKER_UNASSIGNED",
                speaker_name="Chưa phân vai",
                start_time=3.0,
                end_time=None,
                text="Doan thoai khong co end_time.",
            ),
        ]
        custom_map = {"SPEAKER_00": "Quản lý A", "SPEAKER_01": "Kỹ sư B"}
        aligned_mock = align_segments_by_overlap(
            mock_asr_segments,
            merged_turns,
            min_overlap_ratio=0.1,
            speaker_mapping=custom_map,
        )
        assert len(aligned_mock) == 4
        assert aligned_mock[0].speaker_id == "SPEAKER_00"
        assert aligned_mock[0].speaker_name == "Quản lý A"
        assert aligned_mock[1].speaker_id == "SPEAKER_01"
        assert aligned_mock[1].speaker_name == "Kỹ sư B"
        assert aligned_mock[2].speaker_id == "SPEAKER_UNASSIGNED"
        assert aligned_mock[2].speaker_name == "Chưa phân vai"
        assert aligned_mock[3].speaker_id == "SPEAKER_UNASSIGNED"
        assert aligned_mock[3].start_time == 3.0
        assert aligned_mock[3].end_time is None
        logger.info("TEST 2 PASSED: Gan nhan overlap va bao toan timestamp None chinh xac.")

        # ------------------------------------------------------------------
        # TEST 3: Tich hop End-to-End voi auto_detect_diarization=True -> SQLite
        # ------------------------------------------------------------------
        logger.info("--- TEST 3: Tu dong phat hien diarization & xu ly process_audio ---")
        shared_asr = PhoWhisperASR()
        svc = MeetingService(db_path=TEST_DB_PATH, asr=shared_asr)

        resolved_off = svc._resolve_diarization_input(
            AUDIO_SAMPLE_PATH,
            diarization_input=None,
            auto_detect_diarization=False,
        )
        assert resolved_off is None, "Khi auto_detect_diarization=False phai tra ve None"

        resolved_on = svc._resolve_diarization_input(
            AUDIO_SAMPLE_PATH,
            diarization_input=None,
            auto_detect_diarization=True,
        )
        assert resolved_on == DIARIZATION_JSON_PATH, (
            f"Khi auto_detect_diarization=True phai tu dong tim thay {DIARIZATION_JSON_PATH}"
        )

        analysis = svc.process_audio(
            audio_path=AUDIO_SAMPLE_PATH,
            title="Họp kiểm thử Alignment Day 4",
            auto_detect_diarization=True,
        )
        assert isinstance(analysis, MeetingAnalysis)
        assert analysis.id is not None
        assert len(analysis.segments) == 8, f"Ky vong 8 segments, nhan duoc {len(analysis.segments)}"

        details = svc.get_meeting_details(analysis.id)
        assert details is not None, "Phai truy xuat duoc cuoc hop tu test SQLite DB"
        db_segments = details.get("segments", [])
        assert len(db_segments) == 8, f"Ky vong 8 segments trong DB, nhan duoc {len(db_segments)}"

        for idx, (seg_row, ref_seg) in enumerate(zip(db_segments, ref_segments)):
            logger.info(
                "   DB Turn %d: [%s - %s] %s (%s): %s",
                idx + 1,
                seg_row["start_time"],
                seg_row["end_time"],
                seg_row["speaker_name"],
                seg_row["speaker_id"],
                seg_row["text"],
            )
            assert seg_row["speaker_id"] == ref_seg["speaker_id"]
            assert seg_row["start_time"] == ref_seg["start_time"]
            assert seg_row["end_time"] == ref_seg["end_time"]
            assert len(seg_row["text"].strip()) > 0
            assert seg_row["speaker_name"] in ("Người nói 1", "Người nói 2")

        tasks = svc.get_all_tasks()
        assert len(tasks) == 0, "Khong duoc sinh task gia khi xu ly audio"
        logger.info("TEST 3 PASSED: Tu dong phat hien diarization va luu 8 turns thanh cong.")

        # ------------------------------------------------------------------
        # TEST 4: Kiem thu update_speaker_names tren SQLite test DB
        # ------------------------------------------------------------------
        logger.info("--- TEST 4: Kiem thu update_speaker_names tren SQLite ---")
        role_mapping = {"SPEAKER_00": "Trưởng nhóm", "SPEAKER_01": "Thành viên"}
        update_ok = svc.update_speaker_names(analysis.id, role_mapping)
        assert update_ok is True, "update_speaker_names phai tra ve True khi cap nhat thanh cong"

        updated_details = svc.get_meeting_details(analysis.id)
        assert updated_details is not None
        updated_segments = updated_details.get("segments", [])
        assert len(updated_segments) == 8

        for idx, seg_row in enumerate(updated_segments):
            expected_name = role_mapping[seg_row["speaker_id"]]
            assert seg_row["speaker_name"] == expected_name, (
                f"Turn {idx + 1} sai speaker_name sau update: "
                f"{seg_row['speaker_name']} != {expected_name}"
            )
        logger.info(
            "TEST 4 PASSED: Cap nhat speaker_name ('Trưởng nhóm', 'Thành viên') "
            "cho toan bo 8 segments trong SQLite thanh cong."
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
    logger.info("HOAN TAT: 100% CAC BAI KIEM THU DAY 4 DEU THANH CONG!")
    logger.info("=" * 72)


if __name__ == "__main__":
    run_day4_tests()

