"""
Cac thanh phan dung chung (Dependencies, Pydantic Schemas, Helpers) cho he thong FastAPI Routers.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config.settings import BASE_DIR
from src.ai.voiceprint import match_speakers_to_profiles
from src.auth.security import hash_password, is_bcrypt_hash
from src.db.connection import get_db_connection, is_postgres_backend
from src.schemas.meeting import SpeakerProfile
from src.schemas.task import TaskPriority, TaskStatus
from src.services.meeting_service import MeetingService

logger = logging.getLogger(__name__)

WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class LoginPayload(BaseModel):
    """Payload dang nhap bang username va password."""
    username: str = Field(..., description="Ten dang nhap")
    password: str = Field(..., description="Mat khau")


class RegisterPayload(BaseModel):
    """Payload dang ky tai khoan thanh vien moi."""
    username: str = Field(..., description="Ten dang nhap")
    email: Optional[str] = Field(default="", description="Email")
    password: str = Field(..., description="Mat khau")


class SpeakerUpdatePayload(BaseModel):
    """Payload cap nhat ten/chuc vu nguoi noi trong cuoc hop."""
    speaker_id: Optional[str] = Field(default=None, description="Ma SPEAKER_XX can cap nhat")
    name: Optional[str] = Field(default=None, description="Ho va ten nhan su")
    role: Optional[str] = Field(default="", description="Chuc vu nhan su")
    department: Optional[str] = Field(default="", description="Phong ban")
    save_voiceprint: bool = Field(default=False, description="Trich xuat mau giong tu cuoc hop luu vao Admin DB")
    speaker_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Map cap nhat hang loat {speaker_id: display_label}",
    )


class TaskPatchPayload(BaseModel):
    """Payload cap nhat trang thai hoac chi tiet cong viec (HITL)."""
    confirmation_status: Optional[str] = None
    status: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    assigner: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    priority_reason: Optional[str] = None


# Singleton MeetingService
_service_instance: Optional[MeetingService] = None


def get_service() -> MeetingService:
    global _service_instance
    if _service_instance is None:
        _service_instance = MeetingService()
    return _service_instance


def _set_meeting_owner(meeting_id: int, owner_username: str) -> None:
    """Gan chu so huu (owner_username) cho cuoc hop trong Database."""
    clean_owner = (owner_username or "nguyenhoang").strip().lower() or "nguyenhoang"
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE meetings SET owner_username = ? WHERE id = ?",
            (clean_owner, meeting_id),
        )
        conn.commit()


def _get_meeting_owner(meeting_id: int) -> str:
    """Lay owner_username cua mot cuoc hop trong Database."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT owner_username FROM meetings WHERE id = ?",
            (meeting_id,),
        ).fetchone()
        if row is None:
            return ""
        return str(row["owner_username"] or "").strip().lower()


def _can_user_access_meeting(
    _svc: MeetingService,
    meeting_id: int,
    username: Optional[str],
) -> bool:
    """
    Quy tac co lap lich su duy nhat cho moi tai khoan tren Workspace (/):
    Chi tai khoan da tai cuoc hop len (owner_username == username) moi xem duoc
    lich su cuoc hop va cong viec cua cuoc hop do.
    """
    clean_user = (username or "").strip().lower()
    if not clean_user:
        return False

    return _get_meeting_owner(meeting_id) == clean_user


URGENT_TASK_KEYWORDS = (
    "hôm nay",
    "ngay",
    "gấp",
    "khẩn",
    "ngày mai",
    "sáng mai",
    "chiều mai",
    "trong ngày",
)


def _annotate_and_sort_tasks(task_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cham diem uu tien cong viec dua tren deadline va mo ta:
    - Chua hoan thanh va chua tu khoa khan cap -> is_urgent=True, priority_level=0.
    - Day cac viec chua hoan thanh va khan cap len dau danh sach.
    """
    prio_rank = {
        TaskPriority.URGENT.value: 0,
        TaskPriority.HIGH.value: 1,
        TaskPriority.MEDIUM.value: 2,
        TaskPriority.LOW.value: 3,
    }
    enriched: List[Dict[str, Any]] = []
    for item in task_dicts:
        d = dict(item)
        status_val = str(d.get("status") or "").strip()
        is_done = status_val == TaskStatus.DONE.value
        due_text = str(d.get("due_date") or "").lower()
        desc_text = str(d.get("description") or "").lower()
        title_text = str(d.get("title") or "").lower()
        combined = f"{due_text} {desc_text} {title_text}"
        has_urgent_kw = any(kw in combined for kw in URGENT_TASK_KEYWORDS)
        prio_val = str(d.get("priority") or TaskPriority.MEDIUM.value)

        is_urgent = (not is_done) and (has_urgent_kw or prio_val == TaskPriority.URGENT.value)
        if is_done:
            priority_level = 4
        elif is_urgent:
            priority_level = 0
        else:
            priority_level = prio_rank.get(prio_val, 2)

        d["is_urgent"] = bool(is_urgent)
        d["priority_level"] = priority_level
        enriched.append(d)

    enriched.sort(
        key=lambda t: (
            1 if str(t.get("status") or "").strip() == TaskStatus.DONE.value else 0,
            int(t.get("priority_level", 2)),
            -int(t.get("id") or 0),
        )
    )
    return enriched


def _serialize_profile(p: SpeakerProfile) -> Dict[str, Any]:
    """
    Chuyen doi SpeakerProfile sang dict JSON chuan (khong bao gio tra ve password_hash).
    Dong bo trang thai has_voiceprint: chi True khi co vector voiceprint VA file am thanh
    mau thuc su ton tai tren dia.
    """
    dim = 0
    try:
        arr = json.loads(p.voiceprint_json or "[]")
        if isinstance(arr, list):
            dim = len(arr)
    except Exception:
        dim = 0

    audio_exists = False
    if p.sample_audio_path:
        target_audio = Path(p.sample_audio_path)
        if not target_audio.is_absolute():
            target_audio = BASE_DIR / target_audio
        audio_exists = target_audio.exists() and target_audio.is_file()

    has_vp = dim > 0 and audio_exists
    onboarded = bool(p.onboarding_completed) and (has_vp or bool(p.is_admin))
    return {
        "id": p.id,
        "username": p.username or f"user_{p.id}",
        "email": p.email or "",
        "name": p.name,
        "role": p.role,
        "department": p.department or "",
        "display_label": p.display_label,
        "is_admin": bool(p.is_admin),
        "onboarding_completed": onboarded,
        "has_voiceprint": has_vp,
        "embedding_dim": dim if has_vp else 0,
        "sample_audio_path": p.sample_audio_path if has_vp else None,
        "audio_url": f"/api/admin/speakers/{p.id}/audio" if has_vp else None,
        "created_at": str(p.created_at) if p.created_at is not None else None,
    }


def _sync_recent_meetings_voiceprints(svc: MeetingService, limit: int = 5) -> None:
    """Tu dong doi sanh cac ho so da co voiceprint voi cac cuoc hop gan nhat."""
    all_profiles = [
        p for p in svc.list_speaker_profiles()
        if p.onboarding_completed and p.voiceprint_json and p.voiceprint_json != "[]"
    ]
    if not all_profiles:
        return
    recent = svc.get_recent_meetings(limit=limit)
    for m in recent:
        mid = m["id"]
        det = svc.get_meeting_details(mid)
        if det and det.get("segments"):
            matched = match_speakers_to_profiles(
                audio_path=svc.get_meeting_audio_path(mid),
                segments=det["segments"],
                profiles=all_profiles,
            )
            if matched:
                svc.update_speaker_names(
                    mid,
                    {sid: info["display_label"] for sid, info in matched.items()},
                )


def ensure_default_voiceprints_and_sync(service: Optional[MeetingService] = None) -> None:
    """
    Dam bao tai khoan Admin mac dinh 'nguyenhoang' (mat khau '123456', is_admin=1) luon ton tai
    khi bang speaker_profiles dang trong; khong tu dong tao them du lieu rac vao meetings/tasks.
    """
    svc = service or get_service()
    default_pwd_hash = hash_password("123456")
    profiles = svc.list_speaker_profiles()

    if not profiles:
        try:
            admin_prof = SpeakerProfile(
                username="nguyenhoang",
                email="admin@aimeeting.vn",
                password_hash=default_pwd_hash,
                name="Nguyễn Hoàng",
                role="Quản trị viên",
                department="",
                voiceprint_json="[]",
                sample_audio_path=None,
                is_admin=True,
                onboarding_completed=True,
            )
            svc.speaker_repo.create(admin_prof)
            logger.info("Da khoi tao tai khoan Admin mac dinh 'nguyenhoang' voi mat khau '123456'.")
            profiles = svc.list_speaker_profiles()
        except Exception as exc:
            logger.warning("Khong the khoi tao tai khoan Admin mac dinh: %s", exc)

    for prof in profiles:
        if prof.username == "nguyenhoang" or "nguyễn hoàng" in prof.name.lower():
            if prof.username != "nguyenhoang" or not prof.is_admin or not prof.password_hash:
                svc.speaker_repo.update(
                    profile_id=prof.id,
                    username="nguyenhoang",
                    email=prof.email or "admin@aimeeting.vn",
                    password_hash=prof.password_hash or default_pwd_hash,
                    is_admin=True,
                    onboarding_completed=True,
                )

    if not is_postgres_backend():
        _sync_recent_meetings_voiceprints(svc, limit=5)
