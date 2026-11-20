"""
Module quan ly cau hinh tap trung cho ung dung AI Meeting Intelligence.
Doc cac bien moi truong tu file .env neu co.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Xac dinh duong dan goc cua du an (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Nap file .env tu thu muc goc
load_dotenv(BASE_DIR / ".env")

# Cau hinh chung
APP_NAME = "AI Meeting Intelligence & Task Management System"
APP_VERSION = "2.3.0"

# Model Gemini uu tien (loai bo tien to 'models/' hoac dau nhay thua neu co trong .env)
GEMINI_MODEL = (
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    .strip()
    .strip("\"'")
    .removeprefix("models/")
    .strip()
    or "gemini-3.5-flash-lite"
)

# Cau hinh Bao mat Xac thuc JWT (co gia tri mac dinh an toan cho moi truong local)
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "aimeeting-local-dev-jwt-secret-key-2026-safe-default",
).strip() or "aimeeting-local-dev-jwt-secret-key-2026-safe-default"
JWT_ALGORITHM = "HS256"
try:
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
except ValueError:
    JWT_EXPIRE_MINUTES = 1440

# Cau hinh thu muc luu tru du lieu
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
UPLOADS_DIR = DATA_DIR / "uploads"
VOICEPRINTS_DIR = DATA_DIR / "voiceprints"
DB_PATH = DATA_DIR / "meetings.db"

# Dam bao cac thu muc ton tai khi ung dung khoi chay
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VOICEPRINTS_DIR.mkdir(parents=True, exist_ok=True)
