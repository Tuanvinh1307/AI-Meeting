"""
Script kiểm thử tự động toàn diện cho:
- Pydantic Schemas (MeetingAnalysis, TaskItem, TranscriptSegment)
- SQLite Database & Migrations
- Repository Layer (MeetingRepository, TaskRepository)
- Service Layer (MeetingService)
"""
from src.services.meeting_service import MeetingService
from src.schemas.task import TaskConfirmationStatus, TaskStatus

def run_tests():
    print("=== 1. KHỞI TẠO SERVICE LAYER ===")
    svc = MeetingService()

    print("\n=== 2. TEST TẠO CUỘC HỌP & TASKS MỚI (CREATE) ===")
    analysis = svc.process_transcript(
        "Nội dung kiểm thử phân tích và trích xuất dữ liệu.",
        title="Họp thử nghiệm Repository & Schemas mới"
    )
    print(f"-> Tạo thành công Meeting ID: {analysis.id}, Tiêu đề: {analysis.title}")
    assert analysis.id is not None, "Meeting ID không được là None"
    assert len(analysis.tasks) == 2, "Phải có 2 tasks"
    assert len(analysis.segments) == 3, "Phải có 3 segments"

    print("\n=== 3. TEST ĐỌC CHI TIẾT CUỘC HỌP KÈM SEGMENTS (READ) ===")
    details = svc.get_meeting_details(analysis.id)
    assert details is not None, "Không tìm thấy chi tiết cuộc họp"
    print(f"-> Tiêu đề: {details['title']}")
    print(f"-> Số đoạn thoại (Segments): {len(details['segments'])}")
    for s in details['segments']:
        print(f"   [{s['start_time']}s - {s['end_time']}s] {s['speaker_name']} ({s['speaker_id']}): {s['text']}")
    assert len(details['segments']) == 3

    print("\n=== 4. TEST CẬP NHẬT TÓM TẮT CUỘC HỌP (UPDATE) ===")
    new_summary = "Tóm tắt đã được chỉnh sửa sau rà soát nội bộ."
    new_decisions = ["Quyết định số 1: Tăng cường bảo mật", "Quyết định số 2: Phê duyệt ngân sách"]
    upd_ok = svc.update_meeting_summary(analysis.id, new_summary, new_decisions)
    assert upd_ok, "Cập nhật tóm tắt thất bại"
    updated_details = svc.get_meeting_details(analysis.id)
    print(f"-> Tóm tắt mới: {updated_details['summary']}")
    print(f"-> Quyết định mới: {updated_details['key_decisions']}")
    assert updated_details['summary'] == new_summary
    assert len(updated_details['key_decisions']) == 2

    print("\n=== 5. TEST CÁC TRƯỜNG DỮ LIỆU MỚI CỦA TASK ===")
    first_task = analysis.tasks[0]
    print(f"-> Task ID: {first_task.id}")
    print(f"   - Tiêu đề: {first_task.title}")
    print(f"   - Người giao: {first_task.assigner}")
    print(f"   - Người nhận: {first_task.assignee}")
    print(f"   - Deadline: {first_task.due_date}")
    print(f"   - Độ ưu tiên: {first_task.priority.value}")
    print(f"   - Lý do ưu tiên: {first_task.priority_reason}")
    print(f"   - Câu nói gốc (Bằng chứng): {first_task.source_quote}")
    print(f"   - Timestamp gốc: [{first_task.source_start_time}s - {first_task.source_end_time}s]")
    print(f"   - Trạng thái duyệt: {first_task.confirmation_status.value}")
    print(f"   - Tiến độ thực hiện: {first_task.status.value}")

    assert first_task.assigner == "Trần Giám Đốc"
    assert first_task.confirmation_status == TaskConfirmationStatus.PENDING

    print("\n=== 6. TEST CẬP NHẬT TRẠNG THÁI DUYỆT (CONFIRMATION STATUS UPDATE) ===")
    c_ok = svc.update_task_confirmation(first_task.id, TaskConfirmationStatus.CONFIRMED.value)
    assert c_ok, "Cập nhật trạng thái duyệt thất bại"
    task_after_conf = svc.task_repo.get_by_id(first_task.id)
    print(f"-> Trạng thái duyệt sau khi update: {task_after_conf.confirmation_status.value}")
    assert task_after_conf.confirmation_status == TaskConfirmationStatus.CONFIRMED

    print("\n=== 7. TEST CẬP NHẬT TIẾN ĐỘ CÔNG VIỆC (TASK STATUS UPDATE) ===")
    s_ok = svc.update_task_status(first_task.id, TaskStatus.IN_PROGRESS.value)
    assert s_ok, "Cập nhật tiến độ thất bại"
    task_after_stat = svc.task_repo.get_by_id(first_task.id)
    print(f"-> Tiến độ sau khi update: {task_after_stat.status.value}")
    assert task_after_stat.status == TaskStatus.IN_PROGRESS

    print("\n=== 8. TEST LỌC DANH SÁCH TASK (FILTERING) ===")
    confirmed_tasks = svc.get_all_tasks(confirmation_status=TaskConfirmationStatus.CONFIRMED.value)
    print(f"-> Số task đã duyệt: {len(confirmed_tasks)}")
    assert any(t.id == first_task.id for t in confirmed_tasks)

    print("\n==============================================")
    print(">>> TẤT CẢ CÁC BÀI TEST ĐỀU ĐẠT CHUẨN 100%! <<<")
    print("==============================================")

if __name__ == "__main__":
    run_tests()

