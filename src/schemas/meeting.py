"""
Schema định nghĩa dữ liệu cho Biên bản và Kết quả phân tích cuộc họp.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from src.schemas.task import TaskItem


class TranscriptSegment(BaseModel):
    """
    Đại diện cho 1 đoạn phát biểu của 1 người cụ thể (Speaker Diarization & Audio STT).
    """
    speaker_id: Optional[str] = Field(
        default="SPEAKER_00",
        description="ID định danh kỹ thuật của người nói từ mô hình Diarization (ví dụ: SPEAKER_00, SPEAKER_01)",
    )
    speaker_name: Optional[str] = Field(
        default="Người nói",
        description="Tên thực tế hoặc nhãn hiển thị của người nói (ví dụ: Giám đốc Tuấn, Anh Nam)",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Thời điểm bắt đầu đoạn phát biểu trong file ghi âm (tính theo giây)",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="Thời điểm kết thúc đoạn phát biểu trong file ghi âm (tính theo giây)",
    )
    text: str = Field(
        ...,
        description="Nội dung phát biểu được chuyển hóa thành văn bản",
    )


class MeetingAnalysis(BaseModel):
    """
    Kết quả phân tích hoàn chỉnh của một cuộc họp do hệ thống tạo ra.
    Object này đóng vai trò là Data Contract thống nhất xuyên suốt các tầng:
    AI Core -> Service Layer -> UI (Streamlit) / API (FastAPI) / Database.
    """
    id: Optional[int] = Field(default=None, description="ID định danh cuộc họp trong Database")
    title: str = Field(default="Cuộc họp không có tiêu đề", description="Tiêu đề cuộc họp")
    date: Optional[str] = Field(default=None, description="Ngày diễn ra cuộc họp (YYYY-MM-DD)")
    
    # Văn bản cuộc họp: Thô và Đã làm sạch
    raw_transcript: str = Field(..., description="Toàn bộ văn bản cuộc họp thô ban đầu")
    cleaned_transcript: Optional[str] = Field(
        default=None,
        description="Văn bản đã qua tiền xử lý (lọc từ đệm thừa 'ừm, à', sửa lỗi chính tả ngữ cảnh)",
    )
    
    # Các đoạn thoại chi tiết theo người nói và mốc thời gian
    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="Danh sách các đoạn thoại được gắn nhãn người phát biểu và timestamp",
    )
    
    # Kết quả phân tích cấp cao
    summary: str = Field(default="", description="Bản tóm tắt nội dung chính và diễn biến cuộc họp")
    key_decisions: List[str] = Field(
        default_factory=list,
        description="Các kết luận và quyết định quan trọng đã được chốt trong cuộc họp",
    )
    
    # Danh sách công việc trích xuất được
    tasks: List[TaskItem] = Field(
        default_factory=list,
        description="Danh sách các Action Items / Tasks kèm đầy đủ bối cảnh và người liên quan",
    )
