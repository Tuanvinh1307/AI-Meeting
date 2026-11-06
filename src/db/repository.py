"""
Repository Layer: Tách biệt hoàn toàn các câu lệnh SQL khỏi Service Layer.
Bao gồm:
- MeetingRepository: Quản lý CRUD cho cuộc họp và các đoạn thoại (Transcript Segments).
- TaskRepository: Quản lý CRUD cho công việc (Tasks/Action Items).
"""
import json
from typing import List, Optional, Dict, Any
from src.db.connection import get_db_connection
from src.schemas.task import TaskItem, TaskPriority, TaskStatus, TaskConfirmationStatus
from src.schemas.meeting import MeetingAnalysis, TranscriptSegment


class MeetingRepository:
    """Quản lý các thao tác dữ liệu liên quan đến Cuộc họp."""

    def create(self, meeting: MeetingAnalysis) -> int:
        """
        Lưu một cuộc họp mới vào database, bao gồm cả các đoạn thoại (nếu có).
        Trả về ID của cuộc họp vừa tạo.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
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

            # Lưu các đoạn thoại theo người nói (transcript_segments)
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
            return meeting_id

    def get_by_id(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết một cuộc họp kèm các đoạn thoại."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
            row = cursor.fetchone()
            if not row:
                return None

            meeting_dict = dict(row)
            # Parse key_decisions từ JSON
            try:
                meeting_dict["key_decisions"] = json.loads(meeting_dict.get("key_decisions") or "[]")
            except Exception:
                meeting_dict["key_decisions"] = []

            # Lấy các đoạn thoại đi kèm
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
        """Lấy danh sách các cuộc họp gần đây."""
        with get_db_connection() as conn:
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

    def update_summary(self, meeting_id: int, summary: str, key_decisions: Optional[List[str]] = None) -> bool:
        """Cập nhật phần tóm tắt và quyết định chính của cuộc họp."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
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

    def delete(self, meeting_id: int) -> bool:
        """Xóa một cuộc họp (SQLite tự xóa các task và segment liên quan nhờ ON DELETE CASCADE)."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
            conn.commit()
            return cursor.rowcount > 0


class TaskRepository:
    """Quản lý các thao tác dữ liệu liên quan đến Công việc (Tasks)."""

    def _row_to_task(self, row: Any) -> TaskItem:
        """Hàm trợ giúp chuyển một dòng SQLite Row thành Pydantic TaskItem."""
        return TaskItem(
            id=row["id"],
            meeting_id=row["meeting_id"],
            title=row["title"],
            description=row["description"],
            assigner=row["assigner"] or "Chưa xác định",
            assignee=row["assignee"] or "Chưa chỉ định",
            due_date=row["due_date"],
            priority=TaskPriority(row["priority"]) if row["priority"] in [p.value for p in TaskPriority] else TaskPriority.MEDIUM,
            priority_reason=row["priority_reason"] or "",
            source_quote=row["source_quote"] or "",
            source_start_time=row["source_start_time"],
            source_end_time=row["source_end_time"],
            status=TaskStatus(row["status"]) if row["status"] in [s.value for s in TaskStatus] else TaskStatus.TODO,
            confirmation_status=(
                TaskConfirmationStatus(row["confirmation_status"])
                if row["confirmation_status"] in [c.value for c in TaskConfirmationStatus]
                else TaskConfirmationStatus.PENDING
            ),
        )

    def create(self, meeting_id: int, task: TaskItem) -> int:
        """Tạo một công việc mới liên kết với cuộc họp."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
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

    def create_batch(self, meeting_id: int, tasks: List[TaskItem]) -> List[int]:
        """Tạo hàng loạt công việc và trả về danh sách ID."""
        task_ids = []
        for task in tasks:
            tid = self.create(meeting_id, task)
            task_ids.append(tid)
        return task_ids

    def get_by_id(self, task_id: int) -> Optional[TaskItem]:
        """Lấy một công việc theo ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_task(row)

    def list_by_meeting_id(self, meeting_id: int) -> List[TaskItem]:
        """Lấy tất cả công việc của một cuộc họp cụ thể."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE meeting_id = ? ORDER BY id ASC", (meeting_id,))
            rows = cursor.fetchall()
            return [self._row_to_task(r) for r in rows]

    def list_all(
        self,
        status: Optional[str] = None,
        confirmation_status: Optional[str] = None,
    ) -> List[TaskItem]:
        """Lấy toàn bộ công việc, có thể lọc theo tiến độ thực hiện hoặc trạng thái duyệt."""
        with get_db_connection() as conn:
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
        """Cập nhật tiến độ thực hiện của công việc (Cần làm / Đang làm / Hoàn thành)."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
            conn.commit()
            return cursor.rowcount > 0

    def update_confirmation(self, task_id: int, confirmation_status: str) -> bool:
        """Cập nhật trạng thái xác nhận của người dùng (Chờ duyệt / Đã duyệt / Đã từ chối)."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE tasks SET confirmation_status = ? WHERE id = ?",
                (confirmation_status, task_id),
            )
            conn.commit()
            return cursor.rowcount > 0

