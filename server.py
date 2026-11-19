"""
FastAPI Backend Server cho He thong AI Meeting Intelligence.
Khoi tao ung dung FastAPI, phuc vu giao dien Web SPA (/ va /admin)
va ket noi 3 module API Router doc lap trong src/routers/.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.routers import admin_router, auth_router, meeting_router
from src.routers.deps import (
    LoginPayload,
    RegisterPayload,
    SpeakerUpdatePayload,
    TaskPatchPayload,
    UPLOADS_DIR,
    WEB_DIR,
    _annotate_and_sort_tasks,
    _can_user_access_meeting,
    _get_meeting_owner,
    _serialize_profile,
    _set_meeting_owner,
    _sync_recent_meetings_voiceprints,
    ensure_default_voiceprints_and_sync,
    get_service,
    hash_password,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    svc = get_service()
    ensure_default_voiceprints_and_sync(svc)
    yield


app = FastAPI(
    title="AI Meeting Intelligence Enterprise API",
    description="He thong Boc bang PhoWhisper, Nhan dien Giong noi Nhan su (Voiceprint) & Phan tich Bien ban Cuoc hop Gemini.",
    version="2.3.0",
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
# ĐĂNG KÝ CÁC ROUTER CHỨC NĂNG (AUTH, MEETINGS/TASKS, ADMIN)
# ==============================================================================

app.include_router(auth_router)
app.include_router(meeting_router)
app.include_router(admin_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
