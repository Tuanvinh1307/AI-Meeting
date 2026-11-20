import sys as _sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""
Test Suite: test_refactor_phase1.py
Kiem tra toan dien 8 tieu chi sau dot Refactoring 1:
1. PhoWhisper lazy load.
2. Pipeline duoc tai su dung (caching).
3. ASR tra ve dung cau truc du lieu Pydantic TranscriptSegment.
4. Timestamp None duoc bao toan.
5. TaskRepository.create_batch() luu thanh cong khi tat ca task hop le.
6. TaskRepository.create_batch() rollback toan bo khi 1 task bi loi.
7. MeetingService xu ly duoc audio va luu vao database.
8. Cac chuc nang CRUD cu hoat dong binh thuong.

Luu y an toan:
Su dung co so du lieu kiem thu rieng biet tai data/test_refactor.db.
Tuyet doi khong dung data/meetings.db.
"""

import sys
from pathlib import Path
import numpy as np

from config.settings import BASE_DIR
from src.core.ai.asr.phowhisper import PhoWhisperASR
from src.schemas.meeting import MeetingAnalysis, TranscriptSegment
from src.schemas.task import TaskConfirmationStatus, TaskItem, TaskPriority, TaskStatus
from src.db.connection import get_db_connection, init_db
from src.db.repository import MeetingRepository, TaskRepository
from src.services.meeting_service import MeetingService

TEST_DB_PATH = BASE_DIR / "data" / "test_refactor.db"


def cleanup_test_db() -> None:
    """Xoa file database kiem thu neu ton tai."""
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass


def test_1_lazy_loading() -> None:
    """1. Kiem tra PhoWhisper van lazy load."""
    print("[TEST 1] Kiem tra Lazy Loading cua PhoWhisperASR...")
    asr = PhoWhisperASR(model_id="vinai/PhoWhisper-base")
    assert not asr.is_loaded, "Loi: Model bi nap ngay khi khoi tao thay vi lazy load."
    assert asr._pipeline is None, "Loi: _pipeline khong phai la None khi khoi tao."
    print("-> PASSED: PhoWhisper khoi tao tuc thi, khong nap model vao RAM.")


def test_2_and_3_and_4_asr_transcribe_and_timestamps() -> None:
    """
    2. Pipeline duoc tai su dung.
    3. ASR tra ve dung cau truc du lieu.
    4. Timestamp None duoc bao toan.
    """
    print("[TEST 2, 3, 4] Kiem tra pipeline caching, cau truc segment va bao toan timestamp None...")
    asr = PhoWhisperASR()

    # Gia lap pipeline cua Hugging Face de kiem tra logic trich xuat va caching ma khong can tai model that 500MB
    class MockPipeline:
        def __init__(self):
            self.call_count = 0

        def __call__(self, inputs, return_timestamps=True):
            self.call_count += 1
            return {
                "text": "Xin chào toàn thể phòng ban hôm nay chúng ta họp.",
                "chunks": [
                    {
                        "text": "Xin chào toàn thể phòng ban",
                        "timestamp": (0.0, 3.5),
                    },
                    {
                        "text": "hôm nay chúng ta họp.",
                        "timestamp": (3.5, None),  # Model tra ve None cho moc ket thuc
                    },
                ],
            }

    mock_pipe = MockPipeline()
    asr._pipeline = mock_pipe  # Inject mock pipeline

    # Test Caching: goi load_model khi da co pipeline
    asr.load_model()
    assert asr._pipeline is mock_pipe, "Loi: Pipeline bi ghi de thay vi tai su dung."
    print("-> PASSED [2]: Pipeline duoc tai su dung (caching in-memory thanh cong).")

    # Test 3 & 4: Goi transcribe voi gia lap am thanh
    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 giay audio 16kHz
    full_text, segments = asr.transcribe(
        dummy_audio,
        return_timestamps=True,
        speaker_id="SPEAKER_01",
        speaker_name="Nguyen Van A",
    )

    assert isinstance(full_text, str) and len(full_text) > 0, "Loi: full_text khong hop le."
    assert len(segments) == 2, f"Loi: So luong segment khong dung, mong doi 2, nhan duoc {len(segments)}."

    seg1 = segments[0]
    assert isinstance(seg1, TranscriptSegment), "Loi: seg1 khong phai la Pydantic TranscriptSegment."
    assert seg1.start_time == 0.0
    assert seg1.end_time == 3.5
    assert seg1.text == "Xin chào toàn thể phòng ban"
    assert seg1.speaker_id == "SPEAKER_01"
    print("-> PASSED [3]: Cau truc tra ve la Tuple[str, List[TranscriptSegment]] chuan Pydantic.")

    seg2 = segments[1]
    assert seg2.start_time == 3.5
    assert seg2.end_time is None, f"Loi: end_time bi sua thanh {seg2.end_time} thay vi bao toan None."
    assert seg2.text == "hôm nay chúng ta họp."
    print("-> PASSED [4]: Timestamp None duoc bao toan trung thuc, khong bi tao gia.")


def test_5_create_batch_success() -> None:
    """5. TaskRepository.create_batch() luu thanh cong khi tat ca task hop le."""
    print("[TEST 5] Kiem tra TaskRepository.create_batch() luu thanh cong trong transaction...")
    init_db(TEST_DB_PATH)
    meeting_repo = MeetingRepository(db_path=TEST_DB_PATH)
    task_repo = TaskRepository(db_path=TEST_DB_PATH)

    meeting_id = meeting_repo.create(
        MeetingAnalysis(
            title="Cuoc hop test batch",
            date="2026-11-08",
            raw_transcript="Noi dung kiem thu",
            cleaned_transcript=None,
            segments=[],
            summary="Tom tat",
            key_decisions=[],
            tasks=[],
        )
    )

    tasks = [
        TaskItem(
            title=f"Task hop le {i}",
            description=f"Mo ta {i}",
            assigner="Giam doc",
            assignee="Nhan vien A",
            due_date="2026-11-15",
            priority=TaskPriority.HIGH,
            priority_reason="Uu tien",
            source_quote="Trich dan",
            source_start_time=1.0,
            source_end_time=5.0,
            status=TaskStatus.TODO,
            confirmation_status=TaskConfirmationStatus.PENDING,
        )
        for i in range(1, 4)
    ]

    task_ids = task_repo.create_batch(meeting_id, tasks)
    assert len(task_ids) == 3, f"Loi: So luong task_id tra ve khong dung, nhan duoc {len(task_ids)}."

    db_tasks = task_repo.list_by_meeting_id(meeting_id)
    assert len(db_tasks) == 3, f"Loi: So task trong DB khong dung, mong doi 3, nhan duoc {len(db_tasks)}."
    print("-> PASSED [5]: Luu thanh cong 3/3 tasks trong cung mot transaction.")


def test_6_create_batch_rollback() -> None:
    """6. TaskRepository.create_batch() rollback toan bo khi mot task trong batch that bai."""
    print("[TEST 6] Kiem tra TaskRepository.create_batch() rollback toan bo khi co loi...")
    meeting_repo = MeetingRepository(db_path=TEST_DB_PATH)
    task_repo = TaskRepository(db_path=TEST_DB_PATH)

    meeting_id = meeting_repo.create(
        MeetingAnalysis(
            title="Cuoc hop test rollback",
            date="2026-11-08",
            raw_transcript="Noi dung kiem thu rollback",
            cleaned_transcript=None,
            segments=[],
            summary="Tom tat",
            key_decisions=[],
            tasks=[],
        )
    )

    # Tao batch gom 2 task hop le va 1 task bi loi vi pham NOT NULL constraint cua cot title trong SQLite
    task1 = TaskItem(
        title="Task hop le 1",
        description="Mo ta 1",
        status=TaskStatus.TODO,
        confirmation_status=TaskConfirmationStatus.PENDING,
    )
    task2 = TaskItem(
        title="Task hop le 2",
        description="Mo ta 2",
        status=TaskStatus.TODO,
        confirmation_status=TaskConfirmationStatus.PENDING,
    )
    task3_faulty = TaskItem(
        title="Task hop le 3",
        description="Mo ta 3",
        status=TaskStatus.TODO,
        confirmation_status=TaskConfirmationStatus.PENDING,
    )
    # Co tinh bien title thanh None o cap do du lieu thuc de vi pham SQLite NOT NULL constraint cua tasks.title
    object.__setattr__(task3_faulty, "title", None)

    rollback_occurred = False
    try:
        task_repo.create_batch(meeting_id, [task1, task2, task3_faulty])
    except Exception as e:
        rollback_occurred = True
        print(f"   (Bat duoc loi du kien tu SQLite: {type(e).__name__})")

    assert rollback_occurred, "Loi: create_batch khong raise Exception khi task bi loi."

    # Kiem tra trong DB: Toan bo task phai bi rollback, khong duoc ton tai task1 hay task2 nao
    saved_tasks = task_repo.list_by_meeting_id(meeting_id)
    assert len(saved_tasks) == 0, (
        f"Loi nghiem trong: Rollback that bai! Van con {len(saved_tasks)} task duoc luu do dang."
    )
    print("-> PASSED [6]: Rollback thanh cong 100%! Khong co task nao bi luu do dang khi co loi.")


def test_7_meeting_service_process_audio() -> None:
    """7. MeetingService van xu ly duoc audio."""
    print("[TEST 7] Kiem tra MeetingService.process_audio() voi mock ASR...")
    class MockASR:
        def __init__(self):
            self.model_id = "mock"

        def transcribe(self, audio_path, return_timestamps=True, speaker_id="SPEAKER_00", speaker_name="Nguoi noi"):
            return (
                "Bản ghi thử nghiệm xử lý âm thanh trong MeetingService.",
                [
                    TranscriptSegment(
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        start_time=0.0,
                        end_time=4.2,
                        text="Bản ghi thử nghiệm xử lý âm thanh trong MeetingService.",
                    )
                ],
            )

    service = MeetingService(db_path=TEST_DB_PATH, asr=MockASR())

    # Su dung sample audio san co
    audio_path = BASE_DIR / "data" / "samples" / "meeting_test.wav"
    assert audio_path.exists(), f"Khong tim thay {audio_path}"

    analysis = service.process_audio(audio_path, title="Cuoc hop audio kiem thu")
    assert analysis.id is not None, "Loi: analysis khong co id sau khi luu."
    assert len(analysis.segments) == 1, "Loi: segments khong dung."
    assert analysis.tasks == [], "Loi: tasks phai rong vi chua co AI phan tich."

    # Kiem tra doc lai tu DB
    saved_meeting = service.get_meeting_details(analysis.id)
    assert saved_meeting is not None, "Loi: Khong tim thay cuoc hop trong DB."
    assert saved_meeting["title"] == "Cuoc hop audio kiem thu"
    assert len(saved_meeting["segments"]) == 1
    print("-> PASSED [7]: MeetingService.process_audio() xu ly va luu tru thanh cong.")


def test_8_existing_crud_operations() -> None:
    """8. Cac chuc nang CRUD cu van hoat dong on dinh."""
    print("[TEST 8] Kiem tra cac chuc nang CRUD cu...")
    service = MeetingService(db_path=TEST_DB_PATH)

    # 1. Process transcript
    analysis = service.process_transcript(
        raw_transcript="Cuộc họp thảo luận về báo cáo tài chính quý.",
        title="Cuộc họp CRUD Test",
    )
    meeting_id = analysis.id
    assert meeting_id is not None

    # 2. List recent
    recent = service.get_recent_meetings(limit=5)
    assert any(m["id"] == meeting_id for m in recent), "Loi: Khong tim thay cuoc hop trong danh sach gan day."

    # 3. Get all tasks
    tasks = service.get_all_tasks()
    meeting_tasks = [t for t in tasks if t.meeting_id == meeting_id]
    assert len(meeting_tasks) == 2, f"Loi: Mong doi 2 tasks, nhan duoc {len(meeting_tasks)}."

    # 4. Update task status
    task_to_update = meeting_tasks[0]
    success_status = service.update_task_status(task_to_update.id, "Đang làm")
    assert success_status, "Loi: update_task_status that bai."

    # 5. Update task confirmation
    success_confirm = service.update_task_confirmation(task_to_update.id, "Đã duyệt")
    assert success_confirm, "Loi: update_task_confirmation that bai."

    # 6. Update meeting summary
    success_summary = service.update_meeting_summary(
        meeting_id,
        summary="Ban tom tat moi da cap nhat.",
        key_decisions=["Quyet dinh moi 1"],
    )
    assert success_summary, "Loi: update_meeting_summary that bai."

    details = service.get_meeting_details(meeting_id)
    assert details["summary"] == "Ban tom tat moi da cap nhat."
    assert details["key_decisions"] == ["Quyet dinh moi 1"]

    # 7. Delete meeting (Cascade test)
    delete_success = service.meeting_repo.delete(meeting_id)
    assert delete_success, "Loi: delete meeting that bai."
    assert service.get_meeting_details(meeting_id) is None
    # Tasks va segments phai bi cascade delete
    assert len(service.task_repo.list_by_meeting_id(meeting_id)) == 0

    print("-> PASSED [8]: Tat ca thao tac CRUD (create, read, update, delete, cascade) hoat dong hoan hao.")


def main() -> None:
    print("=" * 70)
    print("BAT DAU KIEM THU REFACTORING DOT 1 (BACKEND QUALITY & DATA INTEGRITY)")
    print(f"Test Database: {TEST_DB_PATH}")
    print("=" * 70)

    cleanup_test_db()

    try:
        test_1_lazy_loading()
        test_2_and_3_and_4_asr_transcribe_and_timestamps()
        test_5_create_batch_success()
        test_6_create_batch_rollback()
        test_7_meeting_service_process_audio()
        test_8_existing_crud_operations()
        print("=" * 70)
        print("KET QUA: 8/8 BAI KIEM THU DAT KET QUA HOAN HAO (100% PASSED)!")
        print("=" * 70)
    finally:
        cleanup_test_db()


if __name__ == "__main__":
    main()

