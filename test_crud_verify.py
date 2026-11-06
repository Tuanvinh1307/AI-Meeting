"""
Script kiểm thử tự động toàn diện (Automated Tests).
ĐẢM BẢO AN TOÀN DỮ LIỆU:
- Sử dụng database riêng biệt (data/test_meetings.db) cho toàn bộ bài test.
- TUYỆT ĐỐI KHÔNG ghi, sửa hoặc xóa dữ liệu trong database chính (data/meetings.db).
- Tự động dọn dẹp file test database sau khi hoàn tất kiểm thử.
"""
import os
import sqlite3
from pathlib import Path
from config.settings import BASE_DIR, DB_PATH
from src.services.meeting_service import MeetingService
from src.schemas.task import TaskConfirmationStatus, TaskStatus

# Đường dẫn file SQLite riêng biệt phục vụ kiểm thử
TEST_DB_PATH = BASE_DIR / "data" / "test_meetings.db"


def _get_db_stats(db_file: Path) -> dict:
    """Lấy số lượng bản ghi meetings và tasks của một database file."""
    if not db_file.exists():
        return {"meetings": 0, "tasks": 0}
    conn = sqlite3.connect(db_file)
    try:
        m_count = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
        t_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        return {"meetings": m_count, "tasks": t_count}
    except Exception:
        return {"meetings": 0, "tasks": 0}
    finally:
        conn.close()


def run_tests():
    print("==================================================================")
    print("      KIỂM THỬ TỰ ĐỘNG - MÔI TRƯỜNG TEST ĐỘC LẬP")
    print("==================================================================")

    # 1. Ghi nhận trạng thái database chính trước khi test
    main_db_before = _get_db_stats(DB_PATH)
    print(f"📊 Trạng thái Database CHÍNH ({DB_PATH.name}) TRƯỚC khi test:")
    print(f"   - Số cuộc họp: {main_db_before['meetings']}")
    print(f"   - Số công việc: {main_db_before['tasks']}")
    print("   -> Cam kết: Các số liệu này sẽ được giữ nguyên 100% sau khi test.")

    # 2. Chuẩn bị file database test riêng (xóa nếu còn sót từ lần chạy trước)
    if TEST_DB_PATH.exists():
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass

    print(f"\n🧪 Khởi tạo Service Layer với TEST Database riêng: {TEST_DB_PATH.name}")
    svc = MeetingService(db_path=TEST_DB_PATH)

    # 3. Test Tạo cuộc họp & Task trên Test DB
    print("\n--- BÀI TEST 1: TẠO CUỘC HỌP & TASKS MỚI (CREATE) ---")
    analysis = svc.process_transcript(
        "Nội dung kiểm thử phân tích và trích xuất dữ liệu trên test database.",
        title="Họp thử nghiệm môi trường Test độc lập"
    )
    print(f"-> Tạo thành công Meeting ID: {analysis.id}, Tiêu đề: {analysis.title}")
    assert analysis.id is not None, "Meeting ID không được là None"
    assert len(analysis.tasks) == 2, "Phải có 2 tasks"
    assert len(analysis.segments) == 3, "Phải có 3 segments"

    # 4. Test Đọc chi tiết cuộc họp kèm segments
    print("\n--- BÀI TEST 2: ĐỌC CHI TIẾT CUỘC HỌP KÈM SEGMENTS (READ) ---")
    details = svc.get_meeting_details(analysis.id)
    assert details is not None, "Không tìm thấy chi tiết cuộc họp"
    print(f"-> Tiêu đề: {details['title']}")
    print(f"-> Số đoạn thoại (Segments): {len(details['segments'])}")
    for s in details['segments']:
        print(f"   [{s['start_time']}s - {s['end_time']}s] {s['speaker_name']} ({s['speaker_id']}): {s['text']}")
    assert len(details['segments']) == 3

    # 5. Test Cập nhật tóm tắt cuộc họp
    print("\n--- BÀI TEST 3: CẬP NHẬT TÓM TẮT CUỘC HỌP (UPDATE) ---")
    new_summary = "Tóm tắt đã được chỉnh sửa trong bài kiểm tra cập nhật."
    new_decisions = ["Quyết định Test 1: Đạt chuẩn", "Quyết định Test 2: Hoàn tất"]
    upd_ok = svc.update_meeting_summary(analysis.id, new_summary, new_decisions)
    assert upd_ok, "Cập nhật tóm tắt thất bại"
    updated_details = svc.get_meeting_details(analysis.id)
    print(f"-> Tóm tắt mới: {updated_details['summary']}")
    print(f"-> Quyết định mới: {updated_details['key_decisions']}")
    assert updated_details['summary'] == new_summary
    assert len(updated_details['key_decisions']) == 2

    # 6. Test Các trường dữ liệu mới của Task
    print("\n--- BÀI TEST 4: KIỂM TRA CÁC TRƯỜNG DỮ LIỆU ĐẦY ĐỦ CỦA TASK ---")
    first_task = analysis.tasks[0]
    print(f"-> Task ID: {first_task.id}")
    print(f"   - Tiêu đề: {first_task.title}")
    print(f"   - Người giao (Assigner): {first_task.assigner}")
    print(f"   - Người nhận (Assignee): {first_task.assignee}")
    print(f"   - Deadline: {first_task.due_date}")
    print(f"   - Độ ưu tiên: {first_task.priority.value}")
    print(f"   - Lý do ưu tiên: {first_task.priority_reason}")
    print(f"   - Bằng chứng gốc (Source quote): {first_task.source_quote}")
    print(f"   - Timestamp gốc: [{first_task.source_start_time}s - {first_task.source_end_time}s]")
    print(f"   - Trạng thái xác nhận: {first_task.confirmation_status.value}")
    print(f"   - Tiến độ thực hiện: {first_task.status.value}")

    assert first_task.assigner == "Trần Giám Đốc"
    assert first_task.confirmation_status == TaskConfirmationStatus.PENDING

    # 7. Test Cập nhật trạng thái duyệt (Confirmation Status)
    print("\n--- BÀI TEST 5: CẬP NHẬT TRẠNG THÁI DUYỆT (CONFIRMATION) ---")
    c_ok = svc.update_task_confirmation(first_task.id, TaskConfirmationStatus.CONFIRMED.value)
    assert c_ok, "Cập nhật trạng thái duyệt thất bại"
    task_after_conf = svc.task_repo.get_by_id(first_task.id)
    print(f"-> Trạng thái duyệt sau khi update: {task_after_conf.confirmation_status.value}")
    assert task_after_conf.confirmation_status == TaskConfirmationStatus.CONFIRMED

    # 8. Test Cập nhật tiến độ thực hiện (Task Status)
    print("\n--- BÀI TEST 6: CẬP NHẬT TIẾN ĐỘ THỰC HIỆN (STATUS) ---")
    s_ok = svc.update_task_status(first_task.id, TaskStatus.IN_PROGRESS.value)
    assert s_ok, "Cập nhật tiến độ thất bại"
    task_after_stat = svc.task_repo.get_by_id(first_task.id)
    print(f"-> Tiến độ sau khi update: {task_after_stat.status.value}")
    assert task_after_stat.status == TaskStatus.IN_PROGRESS

    # 9. Test Lọc công việc
    print("\n--- BÀI TEST 7: LỌC DANH SÁCH CÔNG VIỆC ĐÃ DUYỆT ---")
    confirmed_tasks = svc.get_all_tasks(confirmation_status=TaskConfirmationStatus.CONFIRMED.value)
    print(f"-> Số task đã duyệt trong Test DB: {len(confirmed_tasks)}")
    assert any(t.id == first_task.id for t in confirmed_tasks)

    # 10. Dọn dẹp file test database
    if TEST_DB_PATH.exists():
        try:
            os.remove(TEST_DB_PATH)
            print(f"\n🧹 Đã dọn dẹp sạch sẽ file {TEST_DB_PATH.name} sau khi test xong.")
        except Exception:
            pass

    # 11. Đối chiếu lại Database chính để chứng minh không hề bị ảnh hưởng
    print("\n==================================================================")
    print("      ĐỐI CHIẾU KIỂM TRA TOÀN VẸN DATABASE CHÍNH")
    print("==================================================================")
    main_db_after = _get_db_stats(DB_PATH)
    print(f"📊 Trạng thái Database CHÍNH ({DB_PATH.name}) SAU khi test:")
    print(f"   - Số cuộc họp: {main_db_after['meetings']} (Trước test: {main_db_before['meetings']})")
    print(f"   - Số công việc: {main_db_after['tasks']} (Trước test: {main_db_before['tasks']})")

    assert main_db_after['meetings'] == main_db_before['meetings'], "LỖI: Số cuộc họp trong DB chính bị thay đổi!"
    assert main_db_after['tasks'] == main_db_before['tasks'], "LỖI: Số tasks trong DB chính bị thay đổi!"

    print("\n>>> KẾT QUẢ: 100% BÀI TEST PASSED! DATABASE CHÍNH ĐƯỢC BẢO TOÀN NGUYÊN VẸN! <<<")


if __name__ == "__main__":
    run_tests()
