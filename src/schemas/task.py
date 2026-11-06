"""
Schema định nghĩa dữ liệu cho Công việc (Action Item / Task).
Sử dụng Pydantic BaseModel để đảm bảo tính hợp lệ của dữ liệu.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskPriority(str, Enum):
    """Mức độ ưu tiên của công việc do AI hoặc người dùng thiết lập."""
    LOW = "Thấp"
    MEDIUM = "Trung bình"
    HIGH = "Cao"
    URGENT = "Khẩn cấp"


class TaskStatus(str, Enum):
    """
    Trạng thái thực hiện công việc (Tiến độ công việc).
    Phản ánh chu trình làm việc thực tế của nhân sự.
    """
    TODO = "Cần làm"
    IN_PROGRESS = "Đang làm"
    DONE = "Hoàn thành"


class TaskConfirmationStatus(str, Enum):
    """
    Trạng thái xác nhận của người dùng đối với đề xuất của AI (Human-in-the-loop).
    Tách biệt hoàn toàn với tiến độ thực hiện công việc.
    """
    PENDING = "Chờ duyệt"       # AI vừa trích xuất xong, chờ con người kiểm tra
    CONFIRMED = "Đã duyệt"      # Người dùng đồng ý đây là công việc chính thức
    REJECTED = "Đã từ chối"     # AI trích xuất nhầm hoặc người dùng hủy bỏ task này


class TaskItem(BaseModel):
    """
    Đại diện cho 1 công việc được trích xuất từ cuộc họp.
    Hỗ trợ đầy đủ thông tin bối cảnh (Context & Evidence) để giải thích lý do trích xuất.
    """
    id: Optional[int] = Field(default=None, description="ID định danh trong cơ sở dữ liệu")
    meeting_id: Optional[int] = Field(default=None, description="ID cuộc họp liên kết")
    title: str = Field(..., description="Tên hoặc nội dung vắn tắt của công việc")
    description: Optional[str] = Field(default="", description="Mô tả chi tiết yêu cầu công việc")
    
    # Nhân sự liên quan
    assigner: Optional[str] = Field(default="Chưa xác định", description="Người giao việc / Người đưa ra yêu cầu")
    assignee: Optional[str] = Field(default="Chưa chỉ định", description="Người chịu trách nhiệm thực hiện")
    
    # Thời hạn & Ưu tiên
    due_date: Optional[str] = Field(default=None, description="Hạn chót hoàn thành (deadline)")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Mức độ ưu tiên")
    priority_reason: Optional[str] = Field(default="", description="Lý do AI đánh giá mức độ ưu tiên này")
    
    # Bằng chứng gốc từ cuộc họp (Evidence & Traceability)
    source_quote: Optional[str] = Field(default="", description="Câu nói gốc trong cuộc họp làm căn cứ trích xuất task")
    source_start_time: Optional[float] = Field(default=None, description="Mốc thời gian bắt đầu câu nói gốc (giây)")
    source_end_time: Optional[float] = Field(default=None, description="Mốc thời gian kết thúc câu nói gốc (giây)")
    
    # Trạng thái kép: Tiến độ thực hiện & Xác nhận của người dùng
    status: TaskStatus = Field(default=TaskStatus.TODO, description="Tiến độ thực hiện công việc")
    confirmation_status: TaskConfirmationStatus = Field(
        default=TaskConfirmationStatus.PENDING,
        description="Trạng thái xác nhận của người dùng đối với đề xuất của AI",
    )
