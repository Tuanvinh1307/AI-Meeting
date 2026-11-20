"""
Router Dashboard, Cuoc hop, Phat am thanh & Cong viec (/api/dashboard, /api/meetings/*, /api/tasks*).
Toan bo API xac thuc nguoi dung qua JWT dependency (get_current_user / get_current_user_audio),
loai bo hoan toan viec tin tuong tham so ?username= de chong IDOR.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from config.settings import BASE_DIR
from src.core.ai.diarization.voiceprint import match_speakers_to_profiles
from src.core.security import get_current_user, get_current_user_audio
from src.routers.deps import (
    SpeakerUpdatePayload,
    TaskPatchPayload,
    UPLOADS_DIR,
    _annotate_and_sort_tasks,
    _can_user_access_meeting,
    _get_meeting_owner,
    _serialize_profile,
    _set_meeting_owner,
    get_service,
)
from src.schemas.meeting import SpeakerProfile
from src.schemas.task import TaskConfirmationStatus, TaskStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["meetings"])


@router.get("/api/dashboard")
def get_dashboard(
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """Tra ve danh sach cuoc hop va cong viec cua tai khoan xac thuc qua JWT."""
    svc = get_service()
    clean_user = (current_user.username or "").strip().lower()

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

    if current_user.is_admin:
        profiles = svc.list_speaker_profiles()
        serialized_profiles = [_serialize_profile(p) for p in profiles]
        verified_profiles = [p for p in serialized_profiles if p["onboarding_completed"]]
    else:
        serialized_profiles = [_serialize_profile(current_user)]
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


@router.get("/api/meetings/{meeting_id}")
def get_meeting_detail(
    meeting_id: int,
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """Lay chi tiet cuoc hop, segments, tasks theo quyen so huu cua JWT user."""
    svc = get_service()
    details = svc.get_meeting_details(meeting_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Khong tim thay cuoc hop ID={meeting_id}")

    if not _can_user_access_meeting(svc, meeting_id, current_user.username):
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


@router.get("/api/meetings/{meeting_id}/audio")
def get_meeting_audio(
    meeting_id: int,
    current_user: SpeakerProfile = Depends(get_current_user_audio),
) -> FileResponse:
    """
    Phuc vu file am thanh cua cuoc hop.
    Chap nhan JWT qua header Authorization hoac query param ?token=<jwt> cho the <audio>,
    va bat buoc kiem tra quyen so huu cuoc hop truoc khi tra file.
    """
    svc = get_service()
    if not _can_user_access_meeting(svc, meeting_id, current_user.username):
        raise HTTPException(status_code=403, detail="Bạn không có quyền truy cập file ghi âm này.")
    audio_path = svc.get_meeting_audio_path(meeting_id)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Khong tim thay file am thanh cho cuoc hop nay.")
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=audio_path.name,
    )


@router.post("/api/meetings/upload")
async def upload_and_process_meeting(
    title: str = Form("Cuộc họp mới"),
    auto_analyze: bool = Form(True),
    use_sample_audio: bool = Form(False),
    audio: Optional[UploadFile] = File(None),
    diarization: Optional[UploadFile] = File(None),
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Tai len file am thanh (.wav) va gan owner_username theo JWT cua tai khoan dang dang nhap.
    """
    svc = get_service()
    clean_owner = (current_user.username or "").strip().lower()
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

    return get_meeting_detail(analysis.id, current_user=current_user)


@router.post("/api/meetings/{meeting_id}/analyze")
def analyze_existing_meeting(
    meeting_id: int,
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """Tom tat lai noi dung cuoc hop da luu."""
    svc = get_service()
    if not _can_user_access_meeting(svc, meeting_id, current_user.username):
        raise HTTPException(status_code=403, detail="Bạn không có quyền thao tác trên cuộc họp này.")
    try:
        svc.analyze_meeting(meeting_id)
        return get_meeting_detail(meeting_id, current_user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Loi khi phan tich meeting_id=%d: %s", meeting_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/api/meetings/{meeting_id}/speakers")
def update_meeting_speakers(
    meeting_id: int,
    payload: SpeakerUpdatePayload,
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """Cap nhat ten & chuc vu cho SPEAKER_XX trong cuoc hop va dong bo sang summary/tasks."""
    svc = get_service()
    if not _can_user_access_meeting(svc, meeting_id, current_user.username):
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
    return get_meeting_detail(meeting_id, current_user=current_user)


@router.delete("/api/meetings/{meeting_id}")
def delete_meeting_endpoint(
    meeting_id: int,
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """Xoa ho so cuoc hop."""
    svc = get_service()
    if not _can_user_access_meeting(svc, meeting_id, current_user.username):
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


@router.get("/api/tasks")
def list_tasks_endpoint(
    status: Optional[str] = Query(default=None),
    confirmation_status: Optional[str] = Query(default=None),
    meeting_id: Optional[int] = Query(default=None),
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """Lay danh sach cong viec thuoc quyen so huu cua tai khoan dang dang nhap qua JWT."""
    svc = get_service()
    clean_user = (current_user.username or "").strip().lower()
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


@router.patch("/api/tasks/{task_id}")
def patch_task_endpoint(
    task_id: int,
    payload: TaskPatchPayload,
    current_user: SpeakerProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """Cap nhat trang thai phe duyet, tien do hoac chi tiet cong viec (co kiem tra quyen so huu cuoc hop)."""
    svc = get_service()
    existing = svc.task_repo.get_by_id(task_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Khong tim thay cong viec ID={task_id}")

    if existing.meeting_id is not None and not _can_user_access_meeting(
        svc, int(existing.meeting_id), current_user.username
    ):
        raise HTTPException(status_code=403, detail="Bạn không có quyền thao tác trên công việc này.")

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
