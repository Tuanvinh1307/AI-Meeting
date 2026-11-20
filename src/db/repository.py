"""
Repository Layer: Tach biet hoan toan cac thao tac SQL khoi Service Layer.
Ho tro truyen db_path tuy chinh de co lap hoan toan moi truong kiem thu (Testing).
Bao gom:
- MeetingRepository: Quan ly CRUD cho cuoc hop va cac doan thoai (Transcript Segments).
- TaskRepository: Quan ly CRUD cho cong viec (Tasks / Action Items), dam bao tinh toan ven giao dich.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.db.connection import get_db_connection
from src.schemas.meeting import MeetingAnalysis, SpeakerProfile, TranscriptSegment
from src.schemas.task import TaskConfirmationStatus, TaskItem, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class MeetingRepository:
    """Quan ly cac thao tac du lieu lien quan den Cuoc hop."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None) -> None:
        self.db_path = db_path

    def create(self, meeting: MeetingAnalysis) -> int:
        """
        Luu mot cuoc hop moi kem theo cac doan thoai trong cung mot transaction.
        Tra ve ID cua cuoc hop vua tao.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                key_decisions_json = json.dumps(meeting.key_decisions, ensure_ascii=False)

                cursor.execute(
                    """
                    INSERT INTO meetings (
                        title, meeting_date, raw_transcript, cleaned_transcript, summary, key_decisions
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meeting.title,
                        meeting.date,
                        meeting.raw_transcript,
                        meeting.cleaned_transcript,
                        meeting.summary,
                        key_decisions_json,
                    ),
                )
                meeting_id = cursor.lastrowid

                for seg in meeting.segments:
                    cursor.execute(
                        """
                        INSERT INTO transcript_segments (
                            meeting_id, speaker_id, speaker_name, start_time, end_time, text
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            meeting_id,
                            seg.speaker_id,
                            seg.speaker_name,
                            seg.start_time,
                            seg.end_time,
                            seg.text,
                        ),
                    )

                conn.commit()
                logger.debug("Luu cuoc hop thanh cong voi ID=%d, gom %d doan thoai.", meeting_id, len(meeting.segments))
                return meeting_id
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi luu cuoc hop vao co so du lieu: %s", e, exc_info=True)
                raise

    def get_by_id(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Lay thong tin chi tiet mot cuoc hop kem danh sach cac doan thoai."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
            row = cursor.fetchone()
            if not row:
                return None

            meeting_dict = dict(row)
            try:
                meeting_dict["key_decisions"] = json.loads(meeting_dict.get("key_decisions") or "[]")
            except Exception as e:
                logger.warning("Khong the parse key_decisions JSON cho meeting_id=%d: %s", meeting_id, e)
                meeting_dict["key_decisions"] = []

            cursor.execute(
                """
                SELECT speaker_id, speaker_name, start_time, end_time, text
                FROM transcript_segments
                WHERE meeting_id = ?
                ORDER BY id ASC
                """,
                (meeting_id,),
            )
            seg_rows = cursor.fetchall()
            meeting_dict["segments"] = [dict(s) for s in seg_rows]

            return meeting_dict

    def list_all(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Lay danh sach cac cuoc hop gan day nhat."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, meeting_date, summary, created_at
                FROM meetings
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_summary(
        self,
        meeting_id: int,
        summary: str,
        key_decisions: Optional[List[str]] = None,
    ) -> bool:
        """Cap nhat ban tom tat va quyet dinh chinh cua cuoc hop."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                if key_decisions is not None:
                    decisions_json = json.dumps(key_decisions, ensure_ascii=False)
                    cursor.execute(
                        "UPDATE meetings SET summary = ?, key_decisions = ? WHERE id = ?",
                        (summary, decisions_json, meeting_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE meetings SET summary = ? WHERE id = ?",
                        (summary, meeting_id),
                    )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi cap nhat tom tat cho meeting_id=%d: %s", meeting_id, e, exc_info=True)
                raise

    def update_analysis_result(
        self,
        meeting_id: int,
        cleaned_transcript: Optional[str],
        summary: str,
        key_decisions: List[str],
    ) -> bool:
        """
        Cap nhat ket qua phan tich LLM (cleaned_transcript, summary, key_decisions)
        cho mot cuoc hop trong giao dich SQLite an toan.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                decisions_json = json.dumps(key_decisions, ensure_ascii=False)
                cursor.execute(
                    """
                    UPDATE meetings
                    SET cleaned_transcript = ?, summary = ?, key_decisions = ?
                    WHERE id = ?
                    """,
                    (cleaned_transcript, summary, decisions_json, meeting_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Loi khi cap nhat ket qua phan tich cho meeting_id=%d: %s",
                    meeting_id,
                    e,
                    exc_info=True,
                )
                raise

    def update_speaker_names(
        self,
        meeting_id: int,
        speaker_mapping: Dict[str, str],
    ) -> int:
        """
        Cap nhat speaker_name cho cac phan doan thoai cua mot cuoc hop theo speaker_id,
        dong thoi dong bo ten moi vao summary va tasks (assigner, assignee)
        trong mot giao dich SQLite an toan. Tra ve tong so dong duoc cap nhat.
        """
        if not speaker_mapping:
            return 0

        updated_rows = 0
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT DISTINCT speaker_id, speaker_name
                    FROM transcript_segments
                    WHERE meeting_id = ?
                    """,
                    (meeting_id,),
                )
                existing_pairs = cursor.fetchall()
                old_names_by_id: Dict[str, List[str]] = {}
                for row in existing_pairs:
                    sid = row["speaker_id"]
                    sname = row["speaker_name"]
                    if sid and sname:
                        old_names_by_id.setdefault(sid, []).append(sname)

                cursor.execute("SELECT summary, key_decisions FROM meetings WHERE id = ?", (meeting_id,))
                meeting_row = cursor.fetchone()
                current_summary = (meeting_row["summary"] or "") if meeting_row else ""
                current_decisions_raw = (meeting_row["key_decisions"] or "[]") if meeting_row else "[]"

                for speaker_id, speaker_name in speaker_mapping.items():
                    clean_id = str(speaker_id).strip()
                    clean_name = str(speaker_name).strip()
                    if not clean_id or not clean_name:
                        continue

                    cursor.execute(
                        """
                        UPDATE transcript_segments
                        SET speaker_name = ?
                        WHERE meeting_id = ? AND speaker_id = ?
                        """,
                        (clean_name, meeting_id, clean_id),
                    )
                    updated_rows += cursor.rowcount

                    candidate_old_names = set(old_names_by_id.get(clean_id, []))
                    candidate_old_names.add(clean_id)
                    for old_n in list(candidate_old_names):
                        base_old = old_n.split("(")[0].strip()
                        if base_old:
                            candidate_old_names.add(base_old)
                    base_new = clean_name.split("(")[0].strip()
                    if base_new:
                        candidate_old_names.add(base_new)

                    for old_name in candidate_old_names:
                        if not old_name or old_name == clean_name:
                            continue
                        cursor.execute(
                            "UPDATE tasks SET assigner = ? WHERE meeting_id = ? AND assigner = ?",
                            (clean_name, meeting_id, old_name),
                        )
                        cursor.execute(
                            "UPDATE tasks SET assignee = ? WHERE meeting_id = ? AND assignee = ?",
                            (clean_name, meeting_id, old_name),
                        )
                        if old_name in current_summary and clean_name not in current_summary:
                            current_summary = current_summary.replace(old_name, clean_name)
                        if old_name in current_decisions_raw and clean_name not in current_decisions_raw:
                            current_decisions_raw = current_decisions_raw.replace(old_name, clean_name)

                if meeting_row is not None:
                    cursor.execute(
                        "UPDATE meetings SET summary = ?, key_decisions = ? WHERE id = ?",
                        (current_summary, current_decisions_raw, meeting_id),
                    )

                conn.commit()
                logger.debug(
                    "Cap nhat speaker_name cho meeting_id=%d: %d dong thay doi.",
                    meeting_id,
                    updated_rows,
                )
                return updated_rows
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Loi khi cap nhat speaker_name cho meeting_id=%d: %s",
                    meeting_id,
                    e,
                    exc_info=True,
                )
                raise

    def update_segments_text(
        self,
        meeting_id: int,
        cleaned_segments: List[TranscriptSegment],
    ) -> bool:
        """
        Cap nhat noi dung van ban da chuan hoa ngu canh cho cac doan thoai
        thuoc meeting_id trong giao dich SQLite an toan.
        """
        if not cleaned_segments:
            return False

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT id FROM transcript_segments WHERE meeting_id = ? ORDER BY id ASC",
                    (meeting_id,),
                )
                row_ids = [r["id"] for r in cursor.fetchall()]
                if len(row_ids) == len(cleaned_segments):
                    for row_id, seg in zip(row_ids, cleaned_segments):
                        cursor.execute(
                            """
                            UPDATE transcript_segments
                            SET text = ?, speaker_id = ?, speaker_name = ?,
                                start_time = ?, end_time = ?
                            WHERE id = ?
                            """,
                            (
                                seg.text,
                                seg.speaker_id,
                                seg.speaker_name,
                                seg.start_time,
                                seg.end_time,
                                row_id,
                            ),
                        )
                else:
                    cursor.execute(
                        "DELETE FROM transcript_segments WHERE meeting_id = ?",
                        (meeting_id,),
                    )
                    for seg in cleaned_segments:
                        cursor.execute(
                            """
                            INSERT INTO transcript_segments (
                                meeting_id, speaker_id, speaker_name, start_time, end_time, text
                            )
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                meeting_id,
                                seg.speaker_id,
                                seg.speaker_name,
                                seg.start_time,
                                seg.end_time,
                                seg.text,
                            ),
                        )
                conn.commit()
                return True
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Loi khi cap nhat noi dung segments cho meeting_id=%d: %s",
                    meeting_id,
                    e,
                    exc_info=True,
                )
                raise

    def delete(self, meeting_id: int) -> bool:
        """Xoa mot cuoc hop (cascade xoa tasks va segments tuong ung)."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi xoa cuoc hop meeting_id=%d: %s", meeting_id, e, exc_info=True)
                raise


class TaskRepository:
    """Quan ly cac thao tac du lieu lien quan den Cong viec (Tasks)."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None) -> None:
        self.db_path = db_path

    def _row_to_task(self, row: Any) -> TaskItem:
        """Chuyen doi mot dong SQLite Row thanh doi tuong Pydantic TaskItem."""
        return TaskItem(
            id=row["id"],
            meeting_id=row["meeting_id"],
            title=row["title"],
            description=row["description"],
            assigner=row["assigner"] or "Chưa xác định",
            assignee=row["assignee"] or "Chưa chỉ định",
            due_date=row["due_date"],
            priority=(
                TaskPriority(row["priority"])
                if row["priority"] in [p.value for p in TaskPriority]
                else TaskPriority.MEDIUM
            ),
            priority_reason=row["priority_reason"] or "",
            source_quote=row["source_quote"] or "",
            source_start_time=row["source_start_time"],
            source_end_time=row["source_end_time"],
            status=(
                TaskStatus(row["status"])
                if row["status"] in [s.value for s in TaskStatus]
                else TaskStatus.TODO
            ),
            confirmation_status=(
                TaskConfirmationStatus(row["confirmation_status"])
                if row["confirmation_status"] in [c.value for c in TaskConfirmationStatus]
                else TaskConfirmationStatus.PENDING
            ),
        )

    def create(self, meeting_id: int, task: TaskItem) -> int:
        """Tao mot cong viec moi lien ket voi cuoc hop."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO tasks (
                        meeting_id, title, description, assigner, assignee, due_date,
                        priority, priority_reason, source_quote, source_start_time,
                        source_end_time, status, confirmation_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meeting_id,
                        task.title,
                        task.description,
                        task.assigner,
                        task.assignee,
                        task.due_date,
                        task.priority.value,
                        task.priority_reason,
                        task.source_quote,
                        task.source_start_time,
                        task.source_end_time,
                        task.status.value,
                        task.confirmation_status.value,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi tao task cho meeting_id=%d: %s", meeting_id, e, exc_info=True)
                raise

    def create_batch(self, meeting_id: int, tasks: List[TaskItem]) -> List[int]:
        """
        Luu hang loat cong viec trong mot SQLite transaction duy nhat.
        Neu bat ky task nao gap loi, toan bo batch se duoc rollback de bao dam
        tinh toan ven du lieu, khong de lai task ghi do dang trong database.
        """
        if not tasks:
            return []

        task_ids: List[int] = []
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                for task in tasks:
                    cursor.execute(
                        """
                        INSERT INTO tasks (
                            meeting_id, title, description, assigner, assignee, due_date,
                            priority, priority_reason, source_quote, source_start_time,
                            source_end_time, status, confirmation_status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            meeting_id,
                            task.title,
                            task.description,
                            task.assigner,
                            task.assignee,
                            task.due_date,
                            task.priority.value,
                            task.priority_reason,
                            task.source_quote,
                            task.source_start_time,
                            task.source_end_time,
                            task.status.value,
                            task.confirmation_status.value,
                        ),
                    )
                    task_ids.append(cursor.lastrowid)

                conn.commit()
                logger.debug("Luu thanh cong batch %d tasks cho meeting_id=%d.", len(task_ids), meeting_id)
                return task_ids
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Transaction loi trong create_batch cho meeting_id=%d. Da rollback toan bo. Loi: %s",
                    meeting_id,
                    e,
                    exc_info=True,
                )
                raise

    def get_by_id(self, task_id: int) -> Optional[TaskItem]:
        """Lay mot cong viec theo ID."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_task(row)

    def list_by_meeting_id(self, meeting_id: int) -> List[TaskItem]:
        """Lay tat ca cong viec cua mot cuoc hop cu the."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE meeting_id = ? ORDER BY id ASC", (meeting_id,))
            rows = cursor.fetchall()
            return [self._row_to_task(r) for r in rows]

    def list_all(
        self,
        status: Optional[str] = None,
        confirmation_status: Optional[str] = None,
    ) -> List[TaskItem]:
        """Lay toan bo cong viec, co the loc theo tien do hoac trang thai duyet."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM tasks WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)
            if confirmation_status:
                query += " AND confirmation_status = ?"
                params.append(confirmation_status)

            query += " ORDER BY id DESC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_task(r) for r in rows]

    def update_status(self, task_id: int, status: str) -> bool:
        """Cap nhat tien do thuc hien cua cong viec."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi cap nhat status cho task_id=%d: %s", task_id, e, exc_info=True)
                raise

    def replace_meeting_tasks(
        self,
        meeting_id: int,
        tasks: List[TaskItem],
    ) -> List[int]:
        """
        Thay the cac cong viec dang o trang thai 'Chờ duyệt' cua meeting_id bang
        danh sach cong viec moi trong cung mot giao dich nguyen tu (atomic transaction)
        de tranh nhan doi cong viec khi phan tich lai cuoc hop.
        """
        task_ids: List[int] = []
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "DELETE FROM tasks WHERE meeting_id = ? AND confirmation_status = ?",
                    (meeting_id, TaskConfirmationStatus.PENDING.value),
                )
                for task in tasks:
                    cursor.execute(
                        """
                        INSERT INTO tasks (
                            meeting_id, title, description, assigner, assignee, due_date,
                            priority, priority_reason, source_quote, source_start_time,
                            source_end_time, status, confirmation_status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            meeting_id,
                            task.title,
                            task.description,
                            task.assigner,
                            task.assignee,
                            task.due_date,
                            task.priority.value,
                            task.priority_reason,
                            task.source_quote,
                            task.source_start_time,
                            task.source_end_time,
                            task.status.value,
                            task.confirmation_status.value,
                        ),
                    )
                    task_ids.append(cursor.lastrowid)

                conn.commit()
                logger.debug(
                    "Da thay the va luu %d tasks cho meeting_id=%d.",
                    len(task_ids),
                    meeting_id,
                )
                return task_ids
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Loi trong replace_meeting_tasks cho meeting_id=%d: %s",
                    meeting_id,
                    e,
                    exc_info=True,
                )
                raise

    def update_task_details(
        self,
        task_id: int,
        title: str,
        description: str,
        assigner: str,
        assignee: str,
        due_date: Optional[str],
        priority: str,
        priority_reason: str = "",
    ) -> bool:
        """
        Cap nhat chi tiet noi dung cong viec phuc vu co che Human-in-the-loop
        (chinh sua ten, mo ta, nguoi giao, nguoi nhan, han chot, muc do uu tien).
        """
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Tieu de cong viec khong duoc de trong.")

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    UPDATE tasks
                    SET title = ?, description = ?, assigner = ?, assignee = ?,
                        due_date = ?, priority = ?, priority_reason = ?
                    WHERE id = ?
                    """,
                    (
                        clean_title,
                        description.strip(),
                        assigner.strip() or "Chưa xác định",
                        assignee.strip() or "Chưa chỉ định",
                        due_date.strip() if due_date and due_date.strip() else None,
                        priority,
                        priority_reason.strip(),
                        task_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Loi khi cap nhat chi tiet task_id=%d: %s",
                    task_id,
                    e,
                    exc_info=True,
                )
                raise

    def update_confirmation(self, task_id: int, confirmation_status: str) -> bool:
        """Cap nhat trang thai xac nhan cua nguoi dung."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "UPDATE tasks SET confirmation_status = ? WHERE id = ?",
                    (confirmation_status, task_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi cap nhat confirmation cho task_id=%d: %s", task_id, e, exc_info=True)
                raise


class SpeakerProfileRepository:
    """Quan ly ho so nhan su, tai khoan va vector dac trung giong noi (Voiceprint Profiles)."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None) -> None:
        self.db_path = db_path

    @staticmethod
    def _row_to_profile(row: Any) -> SpeakerProfile:
        keys = row.keys() if hasattr(row, "keys") else []
        return SpeakerProfile(
            id=row["id"],
            username=row["username"] if "username" in keys else None,
            email=(row["email"] or "") if "email" in keys else "",
            password_hash=(row["password_hash"] or "") if "password_hash" in keys else "",
            name=row["name"],
            role=row["role"],
            department=row["department"] or "",
            voiceprint_json=row["voiceprint_json"] or "[]",
            sample_audio_path=row["sample_audio_path"],
            is_admin=bool(row["is_admin"]) if "is_admin" in keys and row["is_admin"] is not None else False,
            onboarding_completed=(
                bool(row["onboarding_completed"])
                if "onboarding_completed" in keys and row["onboarding_completed"] is not None
                else True
            ),
            created_at=row["created_at"],
        )

    def create(self, profile: SpeakerProfile) -> int:
        """Luu ho so giong noi nhan su moi vao co so du lieu."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                clean_username = profile.username.strip().lower() if profile.username and profile.username.strip() else None
                cursor.execute(
                    """
                    INSERT INTO speaker_profiles (
                        username, email, password_hash, name, role, department,
                        voiceprint_json, sample_audio_path, is_admin, onboarding_completed
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_username,
                        (profile.email or "").strip(),
                        profile.password_hash or "",
                        profile.name.strip(),
                        profile.role.strip(),
                        (profile.department or "").strip(),
                        profile.voiceprint_json,
                        profile.sample_audio_path,
                        1 if profile.is_admin else 0,
                        1 if profile.onboarding_completed else 0,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi tao speaker_profile: %s", e, exc_info=True)
                raise

    def get_all(self) -> List[SpeakerProfile]:
        """Lay danh sach tat ca nhan su da dang ky trong he thong."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM speaker_profiles ORDER BY id ASC")
            rows = cursor.fetchall()
            return [self._row_to_profile(r) for r in rows]

    def get_by_id(self, profile_id: int) -> Optional[SpeakerProfile]:
        """Lay ho so nhan su theo ID."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM speaker_profiles WHERE id = ?", (profile_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_profile(row)

    def get_by_username(self, username: str) -> Optional[SpeakerProfile]:
        """Lay ho so nhan su theo username (khong phan biet hoa thuong)."""
        clean_user = (username or "").strip().lower()
        if not clean_user:
            return None
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM speaker_profiles WHERE LOWER(username) = LOWER(?)",
                (clean_user,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_profile(row)

    def update(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        department: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        password_hash: Optional[str] = None,
        voiceprint_json: Optional[str] = None,
        sample_audio_path: Optional[str] = None,
        is_admin: Optional[bool] = None,
        onboarding_completed: Optional[bool] = None,
    ) -> Optional[SpeakerProfile]:
        """Cap nhat thong tin ho so nhan su va mau giong."""
        existing = self.get_by_id(profile_id)
        if existing is None:
            return None

        new_username = (
            username.strip().lower()
            if username is not None and username.strip()
            else existing.username
        )
        new_email = email.strip() if email is not None else existing.email
        new_pwd_hash = password_hash if password_hash is not None else existing.password_hash
        new_name = name.strip() if name is not None and name.strip() else existing.name
        new_role = role.strip() if role is not None and role.strip() else existing.role
        new_dept = department.strip() if department is not None else existing.department
        new_vp = voiceprint_json if voiceprint_json is not None else existing.voiceprint_json
        new_audio = sample_audio_path if sample_audio_path is not None else existing.sample_audio_path
        new_admin = is_admin if is_admin is not None else existing.is_admin
        new_onboarded = (
            onboarding_completed
            if onboarding_completed is not None
            else existing.onboarding_completed
        )

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    UPDATE speaker_profiles
                    SET username = ?, email = ?, password_hash = ?, name = ?, role = ?, department = ?,
                        voiceprint_json = ?, sample_audio_path = ?,
                        is_admin = ?, onboarding_completed = ?
                    WHERE id = ?
                    """,
                    (
                        new_username,
                        new_email,
                        new_pwd_hash,
                        new_name,
                        new_role,
                        new_dept,
                        new_vp,
                        new_audio,
                        1 if new_admin else 0,
                        1 if new_onboarded else 0,
                        profile_id,
                    ),
                )
                conn.commit()
                return self.get_by_id(profile_id)
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi cap nhat speaker_profile id=%d: %s", profile_id, e, exc_info=True)
                raise

    def delete(self, profile_id: int) -> bool:
        """Xoa ho so nhan su theo ID."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM speaker_profiles WHERE id = ?", (profile_id,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                logger.error("Loi khi xoa speaker_profile id=%d: %s", profile_id, e, exc_info=True)
                raise


# Alias tuong thich
SpeakerRepository = SpeakerProfileRepository
