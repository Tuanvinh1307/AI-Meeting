"""
Quản lý kết nối và khởi tạo cơ sở dữ liệu SQLite cho hệ thống.
Hỗ trợ cơ chế Migration an toàn và Dependency Injection (truyền db_path tùy chỉnh cho kiểm thử).
"""
import sqlite3
from pathlib import Path
from typing import Set, Optional, Union
from config.settings import DB_PATH


def get_db_connection(db_path: Optional[Union[Path, str]] = None) -> sqlite3.Connection:
    """
    Tạo và trả về kết nối tới file cơ sở dữ liệu SQLite.
    Nếu không truyền db_path, mặc định dùng DB_PATH từ settings (data/meetings.db).
    Thiết lập row_factory = sqlite3.Row để có thể truy xuất dữ liệu theo tên cột như dict.
    Bật PRAGMA foreign_keys = ON để kích hoạt ràng buộc khóa ngoại (ON DELETE CASCADE).
    """
    target_path = Path(db_path) if db_path else DB_PATH
    # Đảm bảo thư mục cha tồn tại
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _get_existing_columns(cursor: sqlite3.Cursor, table_name: str) -> Set[str]:
    """Lấy danh sách các cột hiện có của một bảng trong SQLite."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row["name"] for row in cursor.fetchall()}


def init_db(db_path: Optional[Union[Path, str]] = None) -> None:
    """
    Khởi tạo cấu trúc các bảng dữ liệu nếu chưa tồn tại,
    và tự động chạy migration để bổ sung các cột mới nếu bảng đã có từ trước.
    Có thể truyền db_path riêng cho môi trường kiểm thử (Testing).
    """
    with get_db_connection(db_path=db_path) as conn:
        cursor = conn.cursor()

        # 1. Bảng lưu trữ cuộc họp (meetings)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                meeting_date TEXT,
                raw_transcript TEXT NOT NULL,
                cleaned_transcript TEXT,
                summary TEXT,
                key_decisions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration cho bảng meetings (nếu đã tạo từ trước)
        existing_meeting_cols = _get_existing_columns(cursor, "meetings")
        if "cleaned_transcript" not in existing_meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN cleaned_transcript TEXT;")
        if "key_decisions" not in existing_meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN key_decisions TEXT;")

        # 2. Bảng lưu trữ công việc (tasks)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                assigner TEXT DEFAULT 'Chưa xác định',
                assignee TEXT DEFAULT 'Chưa chỉ định',
                due_date TEXT,
                priority TEXT DEFAULT 'Trung bình',
                priority_reason TEXT DEFAULT '',
                source_quote TEXT DEFAULT '',
                source_start_time REAL,
                source_end_time REAL,
                status TEXT DEFAULT 'Cần làm',
                confirmation_status TEXT DEFAULT 'Chờ duyệt',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            )
        """)

        # Migration an toàn cho bảng tasks (nếu đã có từ trước)
        existing_task_cols = _get_existing_columns(cursor, "tasks")
        migrations = [
            ("assigner", "ALTER TABLE tasks ADD COLUMN assigner TEXT DEFAULT 'Chưa xác định';"),
            ("priority_reason", "ALTER TABLE tasks ADD COLUMN priority_reason TEXT DEFAULT '';"),
            ("source_quote", "ALTER TABLE tasks ADD COLUMN source_quote TEXT DEFAULT '';"),
            ("source_start_time", "ALTER TABLE tasks ADD COLUMN source_start_time REAL;"),
            ("source_end_time", "ALTER TABLE tasks ADD COLUMN source_end_time REAL;"),
            ("confirmation_status", "ALTER TABLE tasks ADD COLUMN confirmation_status TEXT DEFAULT 'Chờ duyệt';"),
        ]
        for col_name, alter_query in migrations:
            if col_name not in existing_task_cols:
                cursor.execute(alter_query)

        # 3. Bảng lưu trữ các đoạn thoại theo người nói (transcript_segments)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                speaker_id TEXT DEFAULT 'SPEAKER_00',
                speaker_name TEXT DEFAULT 'Người nói',
                start_time REAL,
                end_time REAL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
