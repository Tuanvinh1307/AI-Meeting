"""
Package Service Layer: Tầng xử lý nghiệp vụ trung gian.
Đóng vai trò điều phối giữa Giao diện (Streamlit/FastAPI), AI Core và Database.
"""
from src.services.meeting_service import MeetingService

__all__ = ["MeetingService"]

