"""
FastAPI API Routers cho He thong AI Meeting Intelligence.
"""

from src.routers.admin_router import router as admin_router
from src.routers.auth_router import router as auth_router
from src.routers.meeting_router import router as meeting_router

__all__ = [
    "auth_router",
    "meeting_router",
    "admin_router",
]
