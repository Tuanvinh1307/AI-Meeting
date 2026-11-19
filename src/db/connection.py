"""
Quan ly ket noi va khoi tao co so du lieu cho he thong (Ho tro PostgreSQL 17 & SQLite).
- Neu bien moi truong DATABASE_URL bat dau bang postgresql:// hoac postgres:// va khong truyen db_path rieng,
  he thong tu dong dung PostgreSQL qua psycopg2 voi wrapper tuong thich 100% API sqlite3.
- Neu khong co DATABASE_URL hoac truyen db_path tuy chinh (cho unit tests), fallback ve SQLite.
"""

from datetime import date, datetime
import hashlib
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from config.settings import DB_PATH

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _normalize_pg_value(val: Any) -> Any:
    """Chuan hoa kieu thoi gian tu PostgreSQL sang chuoi ISO tuong duong SQLite."""
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(val, date):
        return val.isoformat()
    return val


def _translate_qmark_to_pyformat(sql: str) -> str:
    """
    Chuyen doi placeholder '?' sang '%s' cua psycopg2 mot cach an toan,
    khong anh huong toi dau '?' nam ben trong chuoi nhay don ('...').
    """
    out: List[str] = []
    in_single_quote = False
    i = 0
    length = len(sql)
    while i < length:
        ch = sql[i]
        if ch == "'":
            out.append(ch)
            if in_single_quote and i + 1 < length and sql[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            in_single_quote = not in_single_quote
        elif ch == "?" and not in_single_quote:
            out.append("%s")
        elif ch == "%":
            out.append("%%")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


class PostgresRow:
    """
    Lop boc dong ket qua PostgreSQL tuong thich 100% voi sqlite3.Row:
    ho tro truy cap theo ten cot row['col'], chi so row[0], row.keys(), va dict(row).
    """

    def __init__(self, keys: Sequence[str], values: Sequence[Any]) -> None:
        self._keys: List[str] = list(keys)
        self._values: List[Any] = [_normalize_pg_value(v) for v in values]
        self._mapping: Dict[str, Any] = dict(zip(self._keys, self._values))

    def keys(self) -> List[str]:
        return list(self._keys)

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping.get(key, default)

    def __getitem__(self, key: Union[str, int, slice]) -> Any:
        if isinstance(key, str):
            return self._mapping[key]
        return self._values[key]

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Any:
        return iter(self._values)

    def __repr__(self) -> str:
        return f"PostgresRow({self._mapping!r})"


class PostgresCursorWrapper:
    """Wrapper cho psycopg2 DictCursor tuong thich hoan toan voi sqlite3.Cursor."""

    def __init__(self, raw_cursor: Any) -> None:
        self._cursor = raw_cursor
        self.lastrowid: Optional[int] = None

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    @property
    def description(self) -> Any:
        return self._cursor.description

    def _wrap_row(self, raw_row: Any) -> Optional[PostgresRow]:
        if raw_row is None:
            return None
        if self._cursor.description:
            col_names = [desc[0] for desc in self._cursor.description]
        elif hasattr(raw_row, "keys"):
            col_names = list(raw_row.keys())
        else:
            col_names = [str(idx) for idx in range(len(raw_row))]
        return PostgresRow(col_names, list(raw_row))

    def execute(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
    ) -> "PostgresCursorWrapper":
        self.lastrowid = None
        stripped = sql.strip()
        if stripped.upper().startswith("PRAGMA "):
            return self

        pg_sql = _translate_qmark_to_pyformat(sql)
        is_insert = bool(re.match(r"^\s*INSERT\s+INTO\s+", pg_sql, re.IGNORECASE))
        has_returning = bool(re.search(r"\bRETURNING\b", pg_sql, re.IGNORECASE))
        appended_returning = False

        if is_insert and not has_returning:
            pg_sql = pg_sql.rstrip().rstrip(";") + " RETURNING id"
            appended_returning = True

        if params is not None:
            self._cursor.execute(pg_sql, tuple(params))
        else:
            self._cursor.execute(pg_sql)

        if appended_returning:
            row = self._cursor.fetchone()
            if row is not None:
                self.lastrowid = int(row[0])

        return self

    def executemany(
        self,
        sql: str,
        seq_of_params: Iterable[Sequence[Any]],
    ) -> "PostgresCursorWrapper":
        pg_sql = _translate_qmark_to_pyformat(sql)
        self._cursor.executemany(pg_sql, [tuple(p) for p in seq_of_params])
        return self

    def fetchone(self) -> Optional[PostgresRow]:
        return self._wrap_row(self._cursor.fetchone())

    def fetchall(self) -> List[PostgresRow]:
        rows = self._cursor.fetchall()
        return [wrapped for r in rows if (wrapped := self._wrap_row(r)) is not None]

    def close(self) -> None:
        self._cursor.close()


class PostgresConnectionWrapper:
    """Wrapper cho psycopg2 Connection tuong thich hoan toan voi sqlite3.Connection."""

    def __init__(self, raw_conn: Any) -> None:
        self._conn = raw_conn

    def cursor(self) -> PostgresCursorWrapper:
        return PostgresCursorWrapper(
            self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        )

    def execute(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
    ) -> PostgresCursorWrapper:
        cur = self.cursor()
        return cur.execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PostgresConnectionWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                self.rollback()
            else:
                self.commit()
        finally:
            self.close()


def _get_postgres_url(db_path: Optional[Union[Path, str]] = None) -> Optional[str]:
    """Tra ve chuoi DATABASE_URL neu cau hinh PostgreSQL va khong truyen db_path rieng."""
    if db_path is not None:
        return None
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url.startswith(("postgresql://", "postgres://")):
        return db_url
    return None


def is_postgres_backend(db_path: Optional[Union[Path, str]] = None) -> bool:
    """Kiem tra he thong dang ket noi PostgreSQL hay SQLite."""
    return _get_postgres_url(db_path) is not None


def get_db_connection(
    db_path: Optional[Union[Path, str]] = None,
) -> Union[PostgresConnectionWrapper, sqlite3.Connection]:
    """
    Tao va tra ve ket noi toi co so du lieu (PostgreSQL 17 neu co DATABASE_URL, nguoc lai SQLite).
    """
    pg_url = _get_postgres_url(db_path)
    if pg_url is not None:
        if psycopg2 is None:
            raise ImportError(
                "Chua cai dat psycopg2-binary. Hay chay: pip install psycopg2-binary"
            )
        raw_conn = psycopg2.connect(pg_url)
        return PostgresConnectionWrapper(raw_conn)

    target_path = Path(db_path) if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_path, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 20000;")
    return conn


def _get_existing_columns(cursor: Any, table_name: str, use_postgres: bool) -> Set[str]:
    """Lay danh sach ten cac cot hien co cua mot bang (ho tro ca PostgreSQL va SQLite)."""
    if use_postgres:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = ?
            """,
            (table_name,),
        )
        return {row["column_name"] for row in cursor.fetchall()}

    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row["name"] for row in cursor.fetchall()}


def _seed_default_admin_if_empty(cursor: Any) -> None:
    """
    Chi khoi tao mac dinh duy nhat tai khoan Admin (nguyenhoang / 123456)
    khi bang speaker_profiles dang trong; de trong cac bang meetings, transcript_segments, tasks.
    """
    cursor.execute("SELECT COUNT(*) AS cnt FROM speaker_profiles")
    row = cursor.fetchone()
    count = int(row[0]) if row is not None else 0
    if count > 0:
        return

    default_pwd_hash = hashlib.sha256(b"123456").hexdigest()
    cursor.execute(
        """
        INSERT INTO speaker_profiles (
            username, email, password_hash, name, role, department,
            voiceprint_json, sample_audio_path, is_admin, onboarding_completed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "nguyenhoang",
            "admin@aimeeting.vn",
            default_pwd_hash,
            "Nguyễn Hoàng",
            "Quản trị viên",
            "",
            "[]",
            None,
            1,
            1,
        ),
    )
    logger.info("Da khoi tao tai khoan Admin mac dinh 'nguyenhoang' tren co so du lieu moi.")


def init_db(db_path: Optional[Union[Path, str]] = None) -> None:
    """
    Khoi tao 4 bang du lieu chuan (meetings, transcript_segments, tasks, speaker_profiles)
    va chay migration an toan cho ca PostgreSQL 17 va SQLite.
    """
    use_pg = is_postgres_backend(db_path)
    target_label = "PostgreSQL (DATABASE_URL)" if use_pg else str(Path(db_path) if db_path else DB_PATH)
    logger.info("Khoi tao co so du lieu tai: %s", target_label)

    pk_type = "SERIAL PRIMARY KEY" if use_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    float_type = "DOUBLE PRECISION" if use_pg else "REAL"

    try:
        with get_db_connection(db_path=db_path) as conn:
            cursor = conn.cursor()

            # 1. Bang luu tru cuoc hop (meetings)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS meetings (
                    id {pk_type},
                    title TEXT NOT NULL,
                    meeting_date TEXT,
                    raw_transcript TEXT NOT NULL,
                    cleaned_transcript TEXT,
                    summary TEXT,
                    key_decisions TEXT,
                    owner_username TEXT DEFAULT 'nguyenhoang',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            existing_meeting_cols = _get_existing_columns(cursor, "meetings", use_pg)
            if "cleaned_transcript" not in existing_meeting_cols:
                cursor.execute("ALTER TABLE meetings ADD COLUMN cleaned_transcript TEXT;")
            if "key_decisions" not in existing_meeting_cols:
                cursor.execute("ALTER TABLE meetings ADD COLUMN key_decisions TEXT;")
            if "owner_username" not in existing_meeting_cols:
                cursor.execute("ALTER TABLE meetings ADD COLUMN owner_username TEXT DEFAULT 'nguyenhoang';")

            # 2. Bang luu tru cong viec (tasks)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS tasks (
                    id {pk_type},
                    meeting_id INTEGER REFERENCES meetings(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    description TEXT,
                    assigner TEXT DEFAULT 'Chưa xác định',
                    assignee TEXT DEFAULT 'Chưa chỉ định',
                    due_date TEXT,
                    priority TEXT DEFAULT 'Trung bình',
                    priority_reason TEXT DEFAULT '',
                    source_quote TEXT DEFAULT '',
                    source_start_time {float_type},
                    source_end_time {float_type},
                    status TEXT DEFAULT 'Cần làm',
                    confirmation_status TEXT DEFAULT 'Chờ duyệt',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            existing_task_cols = _get_existing_columns(cursor, "tasks", use_pg)
            task_migrations: List[Tuple[str, str]] = [
                ("assigner", "ALTER TABLE tasks ADD COLUMN assigner TEXT DEFAULT 'Chưa xác định';"),
                ("priority_reason", "ALTER TABLE tasks ADD COLUMN priority_reason TEXT DEFAULT '';"),
                ("source_quote", "ALTER TABLE tasks ADD COLUMN source_quote TEXT DEFAULT '';"),
                ("source_start_time", f"ALTER TABLE tasks ADD COLUMN source_start_time {float_type};"),
                ("source_end_time", f"ALTER TABLE tasks ADD COLUMN source_end_time {float_type};"),
                ("confirmation_status", "ALTER TABLE tasks ADD COLUMN confirmation_status TEXT DEFAULT 'Chờ duyệt';"),
            ]
            for col_name, alter_query in task_migrations:
                if col_name not in existing_task_cols:
                    cursor.execute(alter_query)

            # 3. Bang luu tru cac doan thoai (transcript_segments)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS transcript_segments (
                    id {pk_type},
                    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    speaker_id TEXT DEFAULT 'SPEAKER_00',
                    speaker_name TEXT DEFAULT 'Người nói',
                    start_time {float_type},
                    end_time {float_type},
                    text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. Bang luu tru ho so giong noi nhan su & tai khoan (speaker_profiles)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    id {pk_type},
                    username TEXT,
                    email TEXT DEFAULT '',
                    password_hash TEXT DEFAULT '',
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT DEFAULT '',
                    voiceprint_json TEXT NOT NULL DEFAULT '[]',
                    sample_audio_path TEXT,
                    is_admin INTEGER DEFAULT 0,
                    onboarding_completed INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            existing_speaker_cols = _get_existing_columns(cursor, "speaker_profiles", use_pg)
            speaker_migrations: List[Tuple[str, str]] = [
                ("username", "ALTER TABLE speaker_profiles ADD COLUMN username TEXT;"),
                ("email", "ALTER TABLE speaker_profiles ADD COLUMN email TEXT DEFAULT '';"),
                ("password_hash", "ALTER TABLE speaker_profiles ADD COLUMN password_hash TEXT DEFAULT '';"),
                ("is_admin", "ALTER TABLE speaker_profiles ADD COLUMN is_admin INTEGER DEFAULT 0;"),
                ("onboarding_completed", "ALTER TABLE speaker_profiles ADD COLUMN onboarding_completed INTEGER DEFAULT 1;"),
            ]
            for col_name, alter_query in speaker_migrations:
                if col_name not in existing_speaker_cols:
                    cursor.execute(alter_query)

            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_speaker_profiles_username
                ON speaker_profiles(username)
                WHERE username IS NOT NULL AND username != '';
            """)

            if use_pg:
                _seed_default_admin_if_empty(cursor)

            conn.commit()
            logger.info("Khoi tao co so du lieu va migration hoan tat thanh cong.")
    except Exception as e:
        logger.error("Loi trong qua trinh khoi tao co so du lieu: %s", e, exc_info=True)
        raise
