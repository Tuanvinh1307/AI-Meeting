"""
Module quản lý cấu hình tập trung cho ứng dụng AI Meeting Intelligence.
Đọc các biến môi trường từ file .env nếu có.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Xác định đường dẫn gốc của dự án (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Nạp file .env từ thư mục gốc
load_dotenv(BASE_DIR / ".env")

# Cấu hình chung
APP_NAME = "AI Meeting Intelligence & Task Management System"
APP_VERSION = "0.1.0"

# Cấu hình thư mục lưu trữ dữ liệu
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "meetings.db"

# Đảm bảo các thư mục tồn tại khi ứng dụng khởi chạy
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

