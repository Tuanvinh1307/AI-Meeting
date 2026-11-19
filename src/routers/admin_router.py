"""
Router Quan tri He thong, Nhan su & Mau giong noi (/api/admin/speakers*).
Toan bo API bat buoc xac thuc JWT va quyen Quan tri vien (require_admin / require_admin_audio).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from config.settings import BASE_DIR
from src.auth.security import hash_password, require_admin, require_admin_audio
from src.routers.deps import (
    UPLOADS_DIR,
    _serialize_profile,
    _sync_recent_meetings_voiceprints,
    get_service,
)
from src.schemas.meeting import SpeakerProfile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])


@router.get("/api/admin/speakers")
def list_admin_speakers(
    _admin: SpeakerProfile = Depends(require_admin),
) -> Dict[str, Any]:
    """Lay danh sach toan bo tai khoan nhan su, trang thai Voiceprint va thong ke kho luu tru."""
    svc = get_service()
    profiles = svc.list_speaker_profiles()
    items = [_serialize_profile(p) for p in profiles]

    voiceprints_dir = BASE_DIR / "data" / "voiceprints"
    total_bytes = 0
    if voiceprints_dir.exists():
        for audio_file in voiceprints_dir.iterdir():
            if audio_file.is_file() and audio_file.suffix.lower() in (".wav", ".mp3"):
                try:
                    total_bytes += audio_file.stat().st_size
                except OSError:
                    pass

    verified_count = sum(1 for item in items if item["has_voiceprint"])
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


@router.post("/api/admin/speakers")
async def create_admin_speaker(
    name: str = Form(...),
    role: str = Form(...),
    department: str = Form(""),
    username: Optional[str] = Form(None),
    email: str = Form(""),
    password: Optional[str] = Form(None),
    is_admin: bool = Form(False),
    audio: Optional[UploadFile] = File(None),
    _admin: SpeakerProfile = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Admin tao nhan su moi: Co the thu am / tai file .wav, .mp3 ngay (onboarding_completed=True)
    hoac tao tai khoan truoc (onboarding_completed=False) voi mat khau mac dinh '123456' bam bang bcrypt.
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


@router.put("/api/admin/speakers/{profile_id}")
async def update_admin_speaker(
    profile_id: int,
    request: Request,
    _admin: SpeakerProfile = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Admin cap nhat Ho ten, Chuc vu, Phong ban, Username, Email, Mat khau (bcrypt), Quyen Admin (is_admin)
    hoac cap nhat lai mau giong noi (.wav, .mp3) cua bat ky nguoi dung nao.
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


@router.get("/api/admin/speakers/{profile_id}/audio")
def get_admin_speaker_audio(
    profile_id: int,
    _admin: SpeakerProfile = Depends(require_admin_audio),
) -> FileResponse:
    """
    Phat lai file am thanh mau giong cua nhan su.
    Chap nhan JWT qua header Authorization hoac query param ?token=<jwt> cho the <audio>,
    va bat buoc nguoi goi co quyen Quan tri vien.
    """
    svc = get_service()
    prof = svc.speaker_repo.get_by_id(profile_id)
    if prof is None or not prof.sample_audio_path:
        raise HTTPException(status_code=404, detail="Khong tim thay mau giong cua nhan su nay.")

    target = Path(prof.sample_audio_path)
    if not target.is_absolute():
        target = BASE_DIR / target

    if not target.exists():
        raise HTTPException(status_code=404, detail="File am thanh mau khong con ton tai tren dia.")

    media_type = "audio/mpeg" if target.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(str(target), media_type=media_type, filename=target.name)


@router.delete("/api/admin/speakers/{profile_id}")
def delete_admin_speaker(
    profile_id: int,
    _admin: SpeakerProfile = Depends(require_admin),
) -> Dict[str, Any]:
    """Xoa ho so giong noi nhan su."""
    svc = get_service()
    prof = svc.speaker_repo.get_by_id(profile_id)
    if prof is None:
        raise HTTPException(status_code=404, detail=f"Khong tim thay ho so nhan su ID={profile_id}")

    deleted = svc.delete_speaker_profile(profile_id)
    if prof.sample_audio_path:
        sample_file = Path(prof.sample_audio_path)
        if not sample_file.is_absolute():
            sample_file = BASE_DIR / sample_file
        if sample_file.exists():
            try:
                sample_file.unlink()
            except OSError:
                pass

    return {"deleted": deleted, "profile_id": profile_id}
