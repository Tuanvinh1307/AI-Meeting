"""
Module quan ly cau hinh tap trung cho ung dung AI Meeting Intelligence.
Doc cac bien moi truong tu file .env neu co.
"""

from pathlib import Path
from dotenv import load_dotenv

# Xac dinh duong dan goc cua du an (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Nap file .env tu thu muc goc
load_dotenv(BASE_DIR / ".env")

# Cau hinh chung
APP_NAME = "AI Meeting Intelligence & Task Management System"
APP_VERSION = "2.2.0"

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
