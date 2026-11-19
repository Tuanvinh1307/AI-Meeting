"""
FastAPI Backend Server cho He thong AI Meeting Intelligence.
Cung cap RESTful API ket noi truc tiep voi MeetingService, Xac thuc Mat khau SHA-256,
Phan quyen RBAC, Cong Onboarding Giong noi Bat buoc va Cong Quan tri doc lap (/admin).
"""

from contextlib import asynccontextmanager
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import BASE_DIR
from src.ai.voiceprint import match_speakers_to_profiles
from src.db.connection import get_db_connection, is_postgres_backend
from src.schemas.meeting import SpeakerProfile
from src.schemas.task import TaskConfirmationStatus, TaskPriority, TaskStatus
from src.services.meeting_service import MeetingService

logger = logging.getLogger(__name__)

WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def hash_password(raw_password: str) -> str:
    """Bam mat khau bang SHA-256."""
    return hashlib.sha256((raw_password or "").encode("utf-8")).hexdigest()


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
    """Gan chu so huu (owner_username) cho cuoc hop trong SQLite."""
    clean_owner = (owner_username or "nguyenhoang").strip().lower() or "nguyenhoang"
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE meetings SET owner_username = ? WHERE id = ?",
            (clean_owner, meeting_id),
        )
        conn.commit()


def _get_meeting_owner(meeting_id: int) -> str:
    """Lay owner_username cua mot cuoc hop trong SQLite."""
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
    """Chuyen doi SpeakerProfile sang dict JSON chuan (khong bao gio tra ve password_hash)."""
    dim = 0
    try:
        arr = json.loads(p.voiceprint_json or "[]")
        if isinstance(arr, list):
            dim = len(arr)
    except Exception:
        dim = 0

    has_vp = dim > 0 and bool(p.sample_audio_path)
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
        "embedding_dim": dim,
        "sample_audio_path": p.sample_audio_path,
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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    svc = get_service()
    ensure_default_voiceprints_and_sync(svc)
    yield


app = FastAPI(
    title="AI Meeting Intelligence Enterprise API",
    description="He thong Boc bang PhoWhisper, Nhan dien Giong noi Nhan su (Voiceprint) & Phan tich Bien ban Cuoc hop Gemini.",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ==============================================================================
# HTML PAGE ROUTES (/ VA /admin ĐỘC LẬP)
# ==============================================================================

@app.get("/static/styles.css", include_in_schema=False)
def serve_legacy_css() -> FileResponse:
    """Route tuong thich nguoc cho trinh duyet con luu cache duong dan cu."""
    return FileResponse(str(WEB_DIR / "css" / "styles.css"), media_type="text/css; charset=utf-8")


@app.get("/static/app.js", include_in_schema=False)
def serve_legacy_app_js() -> FileResponse:
    """Route tuong thich nguoc cho trinh duyet con luu cache duong dan cu."""
    return FileResponse(str(WEB_DIR / "js" / "app.js"), media_type="application/javascript; charset=utf-8")


@app.get("/static/admin.js", include_in_schema=False)
def serve_legacy_admin_js() -> FileResponse:
    """Route tuong thich nguoc cho trinh duyet con luu cache duong dan cu."""
    return FileResponse(str(WEB_DIR / "js" / "admin.js"), media_type="application/javascript; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
def serve_workspace_spa() -> FileResponse:
    """Phuc vu Trang Chinh (Meeting Workspace + Auth & Mandatory Onboarding Gate)."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Chua tim thay file giao dien web/index.html")
    return FileResponse(str(index_path), media_type="text/html; charset=utf-8")


@app.get("/admin", response_class=HTMLResponse)
def serve_admin_portal() -> FileResponse:
    """Phuc vu Cong Quan tri He thong & Dinh danh Giong noi doc lap tai /admin."""
    admin_path = WEB_DIR / "admin.html"
    if not admin_path.exists():
        raise HTTPException(status_code=404, detail="Chua tim thay file giao dien web/admin.html")
    return FileResponse(str(admin_path), media_type="text/html; charset=utf-8")


# ==============================================================================
# 3 API XÁC THỰC TÁCH BIỆT: LOGIN, REGISTER & ONBOARDING
# ==============================================================================

@app.post("/api/auth/login")
def auth_login(payload: LoginPayload) -> Dict[str, Any]:
    """
    Dang nhap bang username va password.
    Kiem tra dung mat khau (SHA-256) moi tra ve thong tin user.
    Khong tu dong tao tai khoan moi; neu sai tra ve HTTP 401.
    """
    svc = get_service()
    clean_user = (payload.username or "").strip().lower()
    if not clean_user or not payload.password:
        raise HTTPException(status_code=401, detail="Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")

    profile = svc.speaker_repo.get_by_username(clean_user)
    if profile is None:
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác.")

    expected_hash = profile.password_hash or hash_password("123456")
    if hash_password(payload.password) != expected_hash:
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác.")

    user_dict = _serialize_profile(profile)
    return {
        "user": user_dict,
        "requires_onboarding": not user_dict["onboarding_completed"],
    }


@app.post("/api/auth/register")
def auth_register(payload: RegisterPayload) -> Dict[str, Any]:
    """
    Dang ky tai khoan moi cho thanh vien:
    Nhan username, email, password -> kiem tra trung username, luu password_hash (SHA-256),
    dat onboarding_completed = 0, is_admin = 0.
    """
    svc = get_service()
    raw_user = (payload.username or "").strip().lower()
    clean_user = "".join(ch for ch in raw_user if ch.isalnum() or ch in ("_", ".", "-"))
    if len(clean_user) < 3:
        raise HTTPException(
            status_code=400,
            detail="Tên đăng nhập phải có ít nhất 3 ký tự (chữ cái, số, dấu gạch dưới).",
        )
    if not payload.password or len(payload.password) < 4:
        raise HTTPException(status_code=400, detail="Mật khẩu phải có ít nhất 4 ký tự.")

    existing = svc.speaker_repo.get_by_username(clean_user)
    if existing is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Tên đăng nhập '{clean_user}' đã tồn tại. Vui lòng chọn tên đăng nhập khác.",
        )

    new_profile = SpeakerProfile(
        username=clean_user,
        email=(payload.email or "").strip(),
        password_hash=hash_password(payload.password),
        name=clean_user,
        role="Chưa thiết lập",
        department="",
        voiceprint_json="[]",
        sample_audio_path=None,
        is_admin=False,
        onboarding_completed=False,
    )
    new_id = svc.speaker_repo.create(new_profile)
    created = svc.speaker_repo.get_by_id(new_id) or new_profile
    user_dict = _serialize_profile(created)
    return {
        "user": user_dict,
        "requires_onboarding": True,
    }


@app.post("/api/auth/onboarding")
async def auth_complete_onboarding(
    profile_id: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    name: str = Form(...),
    role: str = Form(...),
    department: str = Form(""),
    audio: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Sau khi dang ky, nguoi dung cap nhat Ho ten, Chuc vu, Phong ban va tai len
    mau giong thu am 10-30s cua chinh tai khoan do de kich hoat onboarding_completed = 1.
    """
    svc = get_service()
    target_profile: Optional[SpeakerProfile] = None
    if profile_id is not None:
        target_profile = svc.speaker_repo.get_by_id(profile_id)
    if target_profile is None and username:
        target_profile = svc.speaker_repo.get_by_username(username)

    if target_profile is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản cần thiết lập hồ sơ.")

    temp_dir = UPLOADS_DIR / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"onboard_{target_profile.id}_{Path(audio.filename or 'voice.wav').name}"

    try:
        raw_bytes = await audio.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="File âm thanh thử giọng bị trống.")
        with open(temp_path, "wb") as f_obj:
            f_obj.write(raw_bytes)

        updated_profile = svc.update_speaker_profile(
            profile_id=target_profile.id,
            name=name.strip(),
            role=role.strip(),
            department=department.strip(),
            audio_path=temp_path,
            onboarding_completed=True,
        )
        _sync_recent_meetings_voiceprints(svc, limit=3)
        user_dict = _serialize_profile(updated_profile)
        return {
            "user": user_dict,
            "onboarding_completed": True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Loi khi hoan tat onboarding: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Không thể xử lý mẫu giọng: {exc}") from exc
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ==============================================================================
# WORKSPACE, MEETINGS & TASKS APIs
# ==============================================================================

@app.get("/api/dashboard")
def get_dashboard(username: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """Tra ve danh sach cuoc hop va cong viec duoc phep truy cap theo tai khoan."""
    svc = get_service()
    clean_user = (username or "").strip().lower()
    if not clean_user:
        return {
            "kpis": {
                "total_meetings": 0,
                "total_tasks": 0,
                "pending_tasks": 0,
                "confirmed_tasks": 0,
                "done_tasks": 0,
                "registered_speakers": 0,
                "total_users": 0,
            },
            "meetings": [],
            "speakers": [],
        }

    all_meetings = svc.get_recent_meetings(limit=100)
    visible_meetings = [
        m for m in all_meetings if _can_user_access_meeting(svc, int(m["id"]), clean_user)
    ]
    visible_meeting_ids = {int(m["id"]) for m in visible_meetings}

    all_tasks = [
        t for t in svc.get_all_tasks()
        if t.meeting_id is not None and int(t.meeting_id) in visible_meeting_ids
    ]
    pending_tasks = [
        t for t in all_tasks if t.confirmation_status == TaskConfirmationStatus.PENDING
    ]
    confirmed_tasks = [
        t for t in all_tasks if t.confirmation_status == TaskConfirmationStatus.CONFIRMED
    ]
    done_tasks = [t for t in all_tasks if t.status == TaskStatus.DONE]

    profile = svc.speaker_repo.get_by_username(clean_user)
    if profile is not None and profile.is_admin:
        profiles = svc.list_speaker_profiles()
        serialized_profiles = [_serialize_profile(p) for p in profiles]
        verified_profiles = [p for p in serialized_profiles if p["onboarding_completed"]]
    else:
        serialized_profiles = [_serialize_profile(profile)] if profile is not None else []
        verified_profiles = [p for p in serialized_profiles if p["onboarding_completed"]]

    return {
        "kpis": {
            "total_meetings": len(visible_meetings),
            "total_tasks": len(all_tasks),
            "pending_tasks": len(pending_tasks),
            "confirmed_tasks": len(confirmed_tasks),
            "done_tasks": len(done_tasks),
            "registered_speakers": len(verified_profiles),
            "total_users": len(serialized_profiles),
        },
        "meetings": visible_meetings,
        "speakers": verified_profiles,
    }


@app.get("/api/meetings/{meeting_id}")
def get_meeting_detail(
    meeting_id: int,
    username: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Lay chi tiet cuoc hop, segments, tasks theo quyen truy cap cua nguoi dung."""
    svc = get_service()
    details = svc.get_meeting_details(meeting_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Khong tim thay cuoc hop ID={meeting_id}")

    if not _can_user_access_meeting(svc, meeting_id, username):
        raise HTTPException(status_code=403, detail="Bạn không có quyền xem cuộc họp này.")

    tasks = svc.get_tasks_by_meeting(meeting_id)
    segments = details.get("segments") or []
    profiles = [
        p for p in svc.list_speaker_profiles()
        if p.onboarding_completed and p.voiceprint_json and p.voiceprint_json != "[]"
    ]
    audio_path = svc.get_meeting_audio_path(meeting_id)

    voiceprint_matches: Dict[str, dict] = {}
    if profiles and segments and audio_path.exists():
        try:
            voiceprint_matches = match_speakers_to_profiles(
                audio_path=audio_path,
                segments=segments,
                profiles=profiles,
            )
        except Exception as exc:
            logger.warning("Khong the tinh voiceprint_matches cho meeting_id=%d: %s", meeting_id, exc)

    speakers_summary: Dict[str, Dict[str, Any]] = {}
    for seg in segments:
        sid = seg.get("speaker_id") or "SPEAKER_00"
        sname = seg.get("speaker_name") or sid
        dur = 0.0
        if seg.get("start_time") is not None and seg.get("end_time") is not None:
            dur = max(0.0, float(seg["end_time"]) - float(seg["start_time"]))
        if sid not in speakers_summary:
            match_info = voiceprint_matches.get(sid)
            speakers_summary[sid] = {
                "speaker_id": sid,
                "speaker_name": sname,
                "utterance_count": 0,
                "total_duration_sec": 0.0,
                "matched_profile": match_info,
            }
        speakers_summary[sid]["utterance_count"] += 1
        speakers_summary[sid]["total_duration_sec"] = round(
            speakers_summary[sid]["total_duration_sec"] + dur, 2
        )

    return {
        "id": details["id"],
        "title": details["title"],
        "meeting_date": details.get("meeting_date"),
        "owner_username": _get_meeting_owner(meeting_id),
        "raw_transcript": details.get("raw_transcript") or "",
        "cleaned_transcript": details.get("cleaned_transcript") or "",
        "summary": details.get("summary") or "",
        "key_decisions": details.get("key_decisions") or [],
        "created_at": details.get("created_at"),
        "audio_url": f"/api/meetings/{meeting_id}/audio",
        "segments": segments,
        "tasks": _annotate_and_sort_tasks([t.model_dump() for t in tasks]),
        "speakers": list(speakers_summary.values()),
        "voiceprint_matches": voiceprint_matches,
    }


@app.get("/api/meetings/{meeting_id}/audio")
def get_meeting_audio(
    meeting_id: int,
    username: Optional[str] = Query(default=None),
) -> FileResponse:
    """Phuc vu file am thanh cua cuoc hop de phat va tua theo giay."""
    svc = get_service()
    if username and not _can_user_access_meeting(svc, meeting_id, username):
        raise HTTPException(status_code=403, detail="Bạn không có quyền truy cập file ghi âm này.")
    audio_path = svc.get_meeting_audio_path(meeting_id)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Khong tim thay file am thanh cho cuoc hop nay.")
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=audio_path.name,
    )


@app.post("/api/meetings/upload")
async def upload_and_process_meeting(
    request: Request,
    title: str = Form("Cuộc họp mới"),
    username: Optional[str] = Form(default=None),
    auto_analyze: bool = Form(True),
    use_sample_audio: bool = Form(False),
    audio: Optional[UploadFile] = File(None),
    diarization: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Tai len file am thanh (.wav) va gan owner_username cho tai khoan dang dang nhap.
    """
    svc = get_service()
    raw_owner = (
        username
        or request.query_params.get("username")
        or request.headers.get("X-Username")
        or ""
    )
    clean_owner = raw_owner.strip().lower()
    if not clean_owner:
        raise HTTPException(status_code=401, detail="Vui lòng đăng nhập trước khi tải lên cuộc họp.")

    temp_dir = UPLOADS_DIR / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    diarization_input: Any = None
    if diarization is not None and diarization.filename:
        try:
            raw_bytes = await diarization.read()
            diarization_input = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"File JSON phan chia nguoi noi khong hop le: {exc}") from exc

    if use_sample_audio or audio is None or not audio.filename:
        audio_source_path = BASE_DIR / "data" / "samples" / "meeting_test.wav"
        if not audio_source_path.exists():
            raise HTTPException(status_code=400, detail="Vui long tai len file am thanh .wav hop le.")
        is_temp_file = False
    else:
        safe_name = Path(audio.filename).name
        audio_source_path = temp_dir / safe_name
        with open(audio_source_path, "wb") as out_f:
            content = await audio.read()
            out_f.write(content)
        is_temp_file = True

    try:
        analysis = svc.process_audio(
            audio_path=audio_source_path,
            title=title,
            diarization_input=diarization_input,
            auto_detect_diarization=(diarization_input is None),
            run_llm_analysis=auto_analyze,
        )
        if analysis.id is not None:
            _set_meeting_owner(analysis.id, clean_owner)
            saved_meeting_wav = UPLOADS_DIR / f"meeting_{analysis.id}.wav"
            shutil.copyfile(audio_source_path, saved_meeting_wav)
    except Exception as exc:
        logger.error("Loi khi xu ly upload cuoc hop: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if is_temp_file and audio_source_path.exists():
            try:
                audio_source_path.unlink()
            except OSError:
                pass

    return get_meeting_detail(analysis.id, username=clean_owner)


@app.post("/api/meetings/{meeting_id}/analyze")
def analyze_existing_meeting(
    meeting_id: int,
    username: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Tom tat lai noi dung cuoc hop da luu."""
    svc = get_service()
    if not _can_user_access_meeting(svc, meeting_id, username):
        raise HTTPException(status_code=403, detail="Bạn không có quyền thao tác trên cuộc họp này.")
    try:
        svc.analyze_meeting(meeting_id)
        return get_meeting_detail(meeting_id, username=username)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Loi khi phan tich meeting_id=%d: %s", meeting_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/meetings/{meeting_id}/speakers")
def update_meeting_speakers(
    meeting_id: int,
    payload: SpeakerUpdatePayload,
    username: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Cap nhat ten & chuc vu cho SPEAKER_XX trong cuoc hop va dong bo sang summary/tasks."""
    svc = get_service()
    if username and not _can_user_access_meeting(svc, meeting_id, username):
        raise HTTPException(status_code=403, detail="Bạn không có quyền chỉnh sửa cuộc họp này.")
    mapping: Dict[str, str] = {}
    if payload.speaker_mapping:
        mapping.update(payload.speaker_mapping)

    if payload.speaker_id and payload.name:
        clean_name = payload.name.strip()
        clean_role = (payload.role or "").strip()
        display_label = f"{clean_name} ({clean_role})" if clean_role else clean_name
        mapping[payload.speaker_id.strip()] = display_label

    if not mapping:
        raise HTTPException(status_code=400, detail="Thieu thong tin speaker_id va ten nguoi noi.")

    svc.update_speaker_names(meeting_id=meeting_id, speaker_mapping=mapping)
    return get_meeting_detail(meeting_id, username=username or _get_meeting_owner(meeting_id))


@app.delete("/api/meetings/{meeting_id}")
def delete_meeting_endpoint(
    meeting_id: int,
    username: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Xoa ho so cuoc hop."""
    svc = get_service()
    if not _can_user_access_meeting(svc, meeting_id, username):
        raise HTTPException(status_code=403, detail="Bạn không có quyền xóa cuộc họp này.")
    deleted = svc.delete_meeting(meeting_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Khong tim thay cuoc hop ID={meeting_id}")
    uploaded_wav = UPLOADS_DIR / f"meeting_{meeting_id}.wav"
    if uploaded_wav.exists():
        try:
            uploaded_wav.unlink()
        except OSError:
            pass
    return {"deleted": True, "meeting_id": meeting_id}


@app.get("/api/tasks")
def list_tasks_endpoint(
    username: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    confirmation_status: Optional[str] = Query(default=None),
    meeting_id: Optional[int] = Query(default=None),
) -> Dict[str, Any]:
    """Lay danh sach cong viec duoc phep xem theo tai khoan dang dang nhap."""
    svc = get_service()
    clean_user = (username or "").strip().lower()
    if not clean_user:
        return {"tasks": [], "total": 0}

    all_meetings = svc.get_recent_meetings(limit=200)
    visible_meetings = [
        m for m in all_meetings if _can_user_access_meeting(svc, int(m["id"]), clean_user)
    ]
    visible_meeting_ids = {int(m["id"]) for m in visible_meetings}

    if meeting_id is not None:
        if int(meeting_id) not in visible_meeting_ids:
            return {"tasks": [], "total": 0}
        tasks = svc.get_tasks_by_meeting(meeting_id)
        if status:
            tasks = [t for t in tasks if t.status.value == status]
        if confirmation_status:
            tasks = [t for t in tasks if t.confirmation_status.value == confirmation_status]
    else:
        tasks = [
            t for t in svc.get_all_tasks(status=status, confirmation_status=confirmation_status)
            if t.meeting_id is not None and int(t.meeting_id) in visible_meeting_ids
        ]

    meetings_map = {int(m["id"]): m["title"] for m in visible_meetings}
    items = []
    for t in tasks:
        d = t.model_dump()
        d["meeting_title"] = meetings_map.get(int(t.meeting_id), f"Cuộc họp #{t.meeting_id}")
        items.append(d)

    sorted_items = _annotate_and_sort_tasks(items)
    return {"tasks": sorted_items, "total": len(sorted_items)}


@app.patch("/api/tasks/{task_id}")
def patch_task_endpoint(task_id: int, payload: TaskPatchPayload) -> Dict[str, Any]:
    """Cap nhat trang thai phe duyet, tien do hoac chinh sua chi tiet cong viec."""
    svc = get_service()
    existing = svc.task_repo.get_by_id(task_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Khong tim thay cong viec ID={task_id}")

    if payload.confirmation_status is not None:
        svc.update_task_confirmation(task_id, payload.confirmation_status)

    if payload.status is not None:
        svc.update_task_status(task_id, payload.status)

    if any(
        val is not None
        for val in (
            payload.title,
            payload.description,
            payload.assigner,
            payload.assignee,
            payload.due_date,
            payload.priority,
            payload.priority_reason,
        )
    ):
        svc.update_task_details(
            task_id=task_id,
            title=payload.title if payload.title is not None else existing.title,
            description=payload.description if payload.description is not None else (existing.description or ""),
            assigner=payload.assigner if payload.assigner is not None else (existing.assigner or "Chưa xác định"),
            assignee=payload.assignee if payload.assignee is not None else (existing.assignee or "Chưa chỉ định"),
            due_date=payload.due_date if payload.due_date is not None else existing.due_date,
            priority=payload.priority if payload.priority is not None else existing.priority.value,
            priority_reason=(
                payload.priority_reason
                if payload.priority_reason is not None
                else (existing.priority_reason or "")
            ),
        )

    updated = svc.task_repo.get_by_id(task_id)
    return {"task": updated.model_dump() if updated else None}


# ==============================================================================
# WEB ADMIN CONSOLE APIs (/api/admin/speakers)
# ==============================================================================

@app.get("/api/admin/speakers")
def list_admin_speakers() -> Dict[str, Any]:
    """Lay danh sach toan bo tai khoan nhan su, trang thai Voiceprint va thong ke kho luu tru."""
    svc = get_service()
    profiles = svc.list_speaker_profiles()
    items = [_serialize_profile(p) for p in profiles]

    voiceprints_dir = BASE_DIR / "data" / "voiceprints"
    total_bytes = 0
    if voiceprints_dir.exists():
        for wav_file in voiceprints_dir.glob("*.wav"):
            try:
                total_bytes += wav_file.stat().st_size
            except OSError:
                pass

    verified_count = sum(1 for item in items if item["onboarding_completed"])
    admin_count = sum(1 for item in items if item["is_admin"])

    return {
        "speakers": items,
        "total": len(items),
        "stats": {
            "total_users": len(items),
            "verified_voiceprints": verified_count,
            "pending_onboarding": len(items) - verified_count,
            "admin_count": admin_count,
            "storage_kb": round(total_bytes / 1024.0, 1),
        },
    }


@app.post("/api/admin/speakers")
async def create_admin_speaker(
    name: str = Form(...),
    role: str = Form(...),
    department: str = Form(""),
    username: Optional[str] = Form(None),
    email: str = Form(""),
    password: Optional[str] = Form(None),
    is_admin: bool = Form(False),
    audio: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Admin tao nhan su moi: Co the thu am / tai file .wav ngay (onboarding_completed=True)
    hoac tao tai khoan truoc (onboarding_completed=False) voi mat khau mac dinh '123456'.
    """
    svc = get_service()
    temp_dir = UPLOADS_DIR / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    pwd_hash = hash_password(password.strip() if password and password.strip() else "123456")

    try:
        if audio is not None and audio.filename:
            raw_bytes = await audio.read()
            if raw_bytes:
                temp_path = temp_dir / f"admin_enroll_{Path(audio.filename).name}"
                with open(temp_path, "wb") as f_obj:
                    f_obj.write(raw_bytes)

        if temp_path is not None:
            profile = svc.enroll_speaker_profile(
                name=name,
                role=role,
                department=department,
                audio_path=temp_path,
                username=username,
                email=email,
                is_admin=is_admin,
            )
            if profile.id:
                profile = svc.speaker_repo.update(profile_id=profile.id, password_hash=pwd_hash) or profile
            _sync_recent_meetings_voiceprints(svc, limit=3)
        else:
            clean_user = (username or "").strip().lower() or None
            new_prof = SpeakerProfile(
                username=clean_user,
                email=email.strip(),
                password_hash=pwd_hash,
                name=name.strip(),
                role=role.strip(),
                department=department.strip(),
                voiceprint_json="[]",
                sample_audio_path=None,
                is_admin=is_admin,
                onboarding_completed=False,
            )
            new_id = svc.speaker_repo.create(new_prof)
            profile = svc.speaker_repo.get_by_id(new_id) or new_prof

        return {"speaker": _serialize_profile(profile)}
    except Exception as exc:
        logger.error("Loi khi Admin tao nhan su: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Khong the tao ho so nhan su: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@app.put("/api/admin/speakers/{profile_id}")
async def update_admin_speaker(profile_id: int, request: Request) -> Dict[str, Any]:
    """
    Admin cap nhat Ho ten, Chuc vu, Phong ban, Username, Email, Mat khau, Quyen Admin (is_admin)
    hoac cap nhat lai mau giong noi (.wav) cua bat ky nguoi dung nao.
    """
    svc = get_service()
    content_type = request.headers.get("content-type", "")
    temp_path: Optional[Path] = None
    password_raw: Optional[str] = None

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            name = form.get("name")
            role = form.get("role")
            department = form.get("department")
            username = form.get("username")
            email = form.get("email")
            password_raw = form.get("password")
            is_admin_raw = form.get("is_admin")
            is_admin = (
                str(is_admin_raw).lower() in ("true", "1", "yes", "on")
                if is_admin_raw is not None
                else None
            )
            audio_obj = form.get("audio")
            if audio_obj is not None and hasattr(audio_obj, "read"):
                raw_bytes = await audio_obj.read()
                if raw_bytes:
                    temp_dir = UPLOADS_DIR / "tmp"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    fname = getattr(audio_obj, "filename", None) or "updated_voice.wav"
                    temp_path = temp_dir / f"admin_upd_{profile_id}_{Path(fname).name}"
                    with open(temp_path, "wb") as f_obj:
                        f_obj.write(raw_bytes)
        else:
            body = await request.json()
            name = body.get("name")
            role = body.get("role")
            department = body.get("department")
            username = body.get("username")
            email = body.get("email")
            password_raw = body.get("password")
            is_admin = body.get("is_admin")

        updated = svc.update_speaker_profile(
            profile_id=profile_id,
            name=str(name) if name is not None else None,
            role=str(role) if role is not None else None,
            department=str(department) if department is not None else None,
            username=str(username) if username is not None else None,
            email=str(email) if email is not None else None,
            is_admin=bool(is_admin) if is_admin is not None else None,
            audio_path=temp_path,
        )
        if password_raw and str(password_raw).strip():
            updated = (
                svc.speaker_repo.update(
                    profile_id=profile_id,
                    password_hash=hash_password(str(password_raw).strip()),
                )
                or updated
            )

        _sync_recent_meetings_voiceprints(svc, limit=3)
        return {"speaker": _serialize_profile(updated)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Loi khi Admin cap nhat speaker_profile id=%d: %s", profile_id, exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Khong the cap nhat ho so: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@app.get("/api/admin/speakers/{profile_id}/audio")
def get_admin_speaker_audio(profile_id: int) -> FileResponse:
    """Phat lai file am thanh mau giong cua nhan su."""
    svc = get_service()
    prof = svc.speaker_repo.get_by_id(profile_id)
    if prof is None or not prof.sample_audio_path:
        raise HTTPException(status_code=404, detail="Khong tim thay mau giong cua nhan su nay.")

    target = Path(prof.sample_audio_path)
    if not target.is_absolute():
        target = BASE_DIR / target

    if not target.exists():
        raise HTTPException(status_code=404, detail="File am thanh mau khong con ton tai tren dia.")

    return FileResponse(str(target), media_type="audio/wav", filename=target.name)


@app.delete("/api/admin/speakers/{profile_id}")
def delete_admin_speaker(profile_id: int) -> Dict[str, Any]:
    """Xoa ho so giong noi nhan su."""
    svc = get_service()
    prof = svc.speaker_repo.get_by_id(profile_id)
    if prof is None:
        raise HTTPException(status_code=404, detail=f"Khong tim thay ho so nhan su ID={profile_id}")

    deleted = svc.delete_speaker_profile(profile_id)
    if prof.sample_audio_path:
        sample_file = BASE_DIR / prof.sample_audio_path
        if sample_file.exists():
            try:
                sample_file.unlink()
            except OSError:
                pass

    return {"deleted": deleted, "profile_id": profile_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
