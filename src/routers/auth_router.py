"""
Router Xac thuc & Onboarding Giong noi (/api/auth/*) su dung JWT + Bcrypt.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.auth.security import (
    authenticate_user_and_upgrade_hash,
    create_access_token,
    get_current_user,
    hash_password,
)
from src.routers.deps import (
    LoginPayload,
    RegisterPayload,
    UPLOADS_DIR,
    _serialize_profile,
    _sync_recent_meetings_voiceprints,
    get_service,
)
from src.schemas.meeting import SpeakerProfile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


@router.post("/api/auth/login")
def auth_login(payload: LoginPayload) -> Dict[str, Any]:
    """
    Dang nhap bang username va password (Bcrypt + tuong thich nguoc plain-text/legacy hash).
    Tra ve JWT access_token va thong tin user.
    """
    svc = get_service()
    clean_user = (payload.username or "").strip().lower()
    if not clean_user or not payload.password:
        raise HTTPException(status_code=401, detail="Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")

    profile = authenticate_user_and_upgrade_hash(
        username=clean_user,
        raw_password=payload.password,
        speaker_repo=svc.speaker_repo,
    )
    if profile is None:
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác.")

    user_dict = _serialize_profile(profile)
    access_token = create_access_token(profile)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_dict,
        "requires_onboarding": not user_dict["onboarding_completed"],
    }


@router.post("/api/auth/register")
def auth_register(payload: RegisterPayload) -> Dict[str, Any]:
    """
    Dang ky tai khoan moi cho thanh vien:
    Bam mat khau bang bcrypt, luu vao DB va cap ngay JWT access_token de thuc hien onboarding.
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
    access_token = create_access_token(created)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_dict,
        "requires_onboarding": True,
    }


@router.post("/api/auth/onboarding")
async def auth_complete_onboarding(
    name: str = Form(...),
    role: str = Form(...),
    department: str = Form(""),
    audio: UploadFile = File(...),
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Sau khi dang ky/dang nhap, nguoi dung xac thuc qua JWT cap nhat Ho ten, Chuc vu
    va tai len mau giong thu am 10-30s cua chinh tai khoan minh de kich hoat onboarding_completed = 1.
    """
    svc = get_service()
    temp_dir = UPLOADS_DIR / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"onboard_{current_user.id}_{Path(audio.filename or 'voice.wav').name}"

    try:
        raw_bytes = await audio.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="File âm thanh thử giọng bị trống.")
        with open(temp_path, "wb") as f_obj:
            f_obj.write(raw_bytes)

        updated_profile = svc.update_speaker_profile(
            profile_id=current_user.id,
            name=name.strip(),
            role=role.strip(),
            department=department.strip(),
            audio_path=temp_path,
            onboarding_completed=True,
        )
        _sync_recent_meetings_voiceprints(svc, limit=3)
        user_dict = _serialize_profile(updated_profile)
        access_token = create_access_token(updated_profile)
        return {
            "access_token": access_token,
            "token_type": "bearer",
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
