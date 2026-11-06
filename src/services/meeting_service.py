"""
MeetingService: Tầng điều phối nghiệp vụ cho ứng dụng.
Quy tắc kiến trúc:
- KHÔNG import Streamlit hay bất kỳ thư viện UI nào.
- KHÔNG viết trực tiếp câu lệnh SQL (toàn bộ SQL được ủy quyền cho Repository Layer).
- Chỉ giao tiếp thông qua Pydantic schemas (MeetingAnalysis, TaskItem, TranscriptSegment).
- Sẵn sàng 100% để tái sử dụng trong FastAPI router endpoints sau này.
"""
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from src.schemas.meeting import MeetingAnalysis, TranscriptSegment
from src.schemas.task import TaskItem, TaskPriority, TaskStatus, TaskConfirmationStatus
from src.db.connection import init_db
from src.db.repository import MeetingRepository, TaskRepository


class MeetingService:
    """Service chịu trách nhiệm xử lý logic nghiệp vụ liên quan đến cuộc họp."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None):
        # Đảm bảo database và migrations đã hoàn tất trên file được chỉ định
        self.db_path = db_path
        init_db(self.db_path)
        self.meeting_repo = MeetingRepository(db_path=self.db_path)
        self.task_repo = TaskRepository(db_path=self.db_path)

    def process_transcript(self, raw_transcript: str, title: str = "Cuộc họp mới") -> MeetingAnalysis:
        """
        Xử lý nội dung cuộc họp thô:
        1. Gọi AI Core (LangGraph) để phân tích (tạm thời trả về Mock Data có cấu trúc chuẩn).
        2. Ủy quyền cho Repository Layer để lưu vào Database.
        3. Trả về Pydantic model MeetingAnalysis chuẩn hoá cho giao diện.
        """
        if not raw_transcript.strip():
            raise ValueError("Nội dung cuộc họp không được để trống.")

        today_str = datetime.now().strftime("%Y-%m-%d")

        # ----------------------------------------------------------------------
        # TODO: Sau này bước này sẽ gọi LangGraph Workflow trong src/ai/agents/
        # Dưới đây là Mock Data có đầy đủ các trường mới theo yêu cầu kiến trúc
        # ----------------------------------------------------------------------
        sample_segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                speaker_name="Trần Giám Đốc",
                start_time=12.5,
                end_time=35.0,
                text="Chào mọi người, tuần này chúng ta cần ưu tiên hoàn thành báo cáo tài chính quý trước.",
            ),
            TranscriptSegment(
                speaker_id="SPEAKER_01",
                speaker_name="Nguyễn Văn A",
                start_time=36.2,
                end_time=58.0,
                text="Tôi sẽ phụ trách tổng hợp toàn bộ số liệu doanh thu và gửi trước thứ Sáu tuần này.",
            ),
            TranscriptSegment(
                speaker_id="SPEAKER_02",
                speaker_name="Trần Thị B",
                start_time=60.0,
                end_time=82.4,
                text="Sau khi cuộc họp kết thúc, tôi sẽ soạn thảo và gửi email biên bản họp cho cả phòng.",
            ),
        ]

        sample_tasks = [
            TaskItem(
                title="Chuẩn bị tài liệu báo cáo tài chính quý",
                description="Tổng hợp số liệu doanh thu và chi phí quý trước để trình ban giám đốc",
                assigner="Trần Giám Đốc",
                assignee="Nguyễn Văn A",
                due_date=today_str,
                priority=TaskPriority.HIGH,
                priority_reason="Ảnh hưởng trực tiếp đến kế hoạch duyệt ngân sách tuần tới",
                source_quote="Tôi sẽ phụ trách tổng hợp toàn bộ số liệu doanh thu và gửi trước thứ Sáu tuần này.",
                source_start_time=36.2,
                source_end_time=58.0,
                status=TaskStatus.TODO,
                confirmation_status=TaskConfirmationStatus.PENDING,
            ),
            TaskItem(
                title="Gửi email biên bản cuộc họp",
                description="Gửi bản tóm tắt nội dung và action items cho toàn bộ thành viên tham gia",
                assigner="Trần Giám Đốc",
                assignee="Trần Thị B",
                due_date=today_str,
                priority=TaskPriority.MEDIUM,
                priority_reason="Đảm bảo mọi người nắm được quyết định và nhiệm vụ sau cuộc họp",
                source_quote="Sau khi cuộc họp kết thúc, tôi sẽ soạn thảo và gửi email biên bản họp cho cả phòng.",
                source_start_time=60.0,
                source_end_time=82.4,
                status=TaskStatus.TODO,
                confirmation_status=TaskConfirmationStatus.PENDING,
            ),
        ]

        analysis = MeetingAnalysis(
            title=title,
            date=today_str,
            raw_transcript=raw_transcript,
            cleaned_transcript="Cuộc họp thảo luận về tiến độ công việc và báo cáo tài chính quý.",
            segments=sample_segments,
            summary="Cuộc họp định kỳ thảo luận về tiến độ công việc. Ban giám đốc yêu cầu hoàn thiện báo cáo tài chính và gửi biên bản họp trong tuần.",
            key_decisions=[
                "Thống nhất hạn chót nộp báo cáo tài chính quý là thứ Sáu",
                "Gửi email biên bản họp cho toàn bộ nhân sự liên quan",
            ],
            tasks=sample_tasks,
        )

        # ----------------------------------------------------------------------
        # Lưu trữ thông qua Repository Layer (Không dùng SQL trực tiếp ở đây)
        # ----------------------------------------------------------------------
        meeting_id = self.meeting_repo.create(analysis)
        analysis.id = meeting_id

        task_ids = self.task_repo.create_batch(meeting_id, analysis.tasks)
        for task, tid in zip(analysis.tasks, task_ids):
            task.id = tid
            task.meeting_id = meeting_id

        return analysis

    def get_recent_meetings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Lấy danh sách các cuộc họp gần đây nhất."""
        return self.meeting_repo.list_all(limit=limit)

    def get_meeting_details(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Lấy chi tiết một cuộc họp kèm các đoạn thoại."""
        return self.meeting_repo.get_by_id(meeting_id)

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        confirmation_status: Optional[str] = None,
    ) -> List[TaskItem]:
        """Lấy toàn bộ danh sách công việc có lọc theo trạng thái."""
        return self.task_repo.list_all(status=status, confirmation_status=confirmation_status)

    def update_task_status(self, task_id: int, status: str) -> bool:
        """Cập nhật tiến độ công việc (Cần làm / Đang làm / Hoàn thành)."""
        return self.task_repo.update_status(task_id, status)

    def update_task_confirmation(self, task_id: int, confirmation_status: str) -> bool:
        """Cập nhật trạng thái duyệt của người dùng (Chờ duyệt / Đã duyệt / Đã từ chối)."""
        return self.task_repo.update_confirmation(task_id, confirmation_status)

    def update_meeting_summary(self, meeting_id: int, summary: str, key_decisions: Optional[List[str]] = None) -> bool:
        """Cập nhật tóm tắt và quyết định chính."""
        return self.meeting_repo.update_summary(meeting_id, summary, key_decisions)
