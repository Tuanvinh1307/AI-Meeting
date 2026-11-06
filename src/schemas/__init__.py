"""
Package schemas: Định nghĩa các Data Contracts bằng Pydantic.
Dùng để chuẩn hoá cấu trúc dữ liệu truyền giữa UI, Service, AI Core và Database.
"""
from src.schemas.task import TaskItem, TaskPriority, TaskStatus, TaskConfirmationStatus
from src.schemas.meeting import MeetingAnalysis, TranscriptSegment

__all__ = [
    "TaskItem",
    "TaskPriority",
    "TaskStatus",
    "TaskConfirmationStatus",
    "MeetingAnalysis",
    "TranscriptSegment",
]
