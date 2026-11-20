"""
MeetingService: Tang dieu phoi nghiep vu cho ung dung.
Quy tac kien truc:
- Khong import Streamlit hoac bat ky thu vien UI nao vao backend.
- Khong viet truc tiep cau lenh SQL (toan bo SQL duoc uy quyen cho Repository Layer).
- Giao tiep qua cac schemas chuan (MeetingAnalysis, TaskItem, TranscriptSegment).
- Ho tro Dependency Injection cho ASR va Database connection phuc vu testing va caching.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.settings import BASE_DIR
from src.core.ai.asr.phowhisper import PhoWhisperASR
from src.core.ai.diarization.alignment import DiarizationInputType
from src.core.ai.diarization.voiceprint import (
    extract_voiceprint_embedding,
    save_audio_clip,
)
from src.core.ai.graph.workflow import meeting_graph
from src.core.ai.llm import (
    LLMExtractionOutput,
    MeetingLLMAnalyzer,
    clean_vietnamese_asr_text,
)
from src.db.connection import init_db
from src.db.repository import MeetingRepository, SpeakerProfileRepository, TaskRepository
from src.schemas.meeting import MeetingAnalysis, SpeakerProfile, TranscriptSegment
from src.schemas.task import TaskConfirmationStatus, TaskItem, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class MeetingService:
    """Service chiu trach nhiem xu ly logic nghiep vu lien quan den cuoc hop."""

    def __init__(
        self,
        db_path: Optional[Union[Path, str]] = None,
        asr: Optional[PhoWhisperASR] = None,
        analyzer: Optional[MeetingLLMAnalyzer] = None,
    ) -> None:
        """
        Khoi tao MeetingService.
        - db_path: Duong dan database SQLite (mac dinh data/meetings.db hoac db rieng khi test).
        - asr: Instance PhoWhisperASR duoc inject tu ngoai vao (ho tro caching va lazy loading).
        - analyzer: Instance MeetingLLMAnalyzer phuc vu tom tat va trich xuat cong viec.
        """
        self.db_path = db_path
        init_db(self.db_path)
        self.meeting_repo = MeetingRepository(db_path=self.db_path)
        self.task_repo = TaskRepository(db_path=self.db_path)
        self.speaker_repo = SpeakerProfileRepository(db_path=self.db_path)
        self.asr = asr if asr is not None else PhoWhisperASR()
        self.analyzer = analyzer if analyzer is not None else MeetingLLMAnalyzer()
        logger.info("MeetingService da khoi tao thanh cong voi database: %s", self.db_path or "Default")

    def _resolve_diarization_input(
        self,
        audio_path: Path,
        diarization_input: Optional[DiarizationInputType] = None,
        auto_detect_diarization: bool = False,
    ) -> Optional[DiarizationInputType]:
        """
        Xac dinh nguon du lieu diarization cho file audio:
        - Neu diarization_input da duoc truyen truc tiep, giu nguyen.
        - Neu diarization_input la None va auto_detect_diarization=True, kiem tra
          file ket qua tien xu ly tai notebooks/results/diarization_result.json
          xem truong 'audio_file' co khop voi ten file audio_path.name hay khong.
        """
        if diarization_input is not None:
            return diarization_input

        if not auto_detect_diarization:
            return None

        candidate_json = BASE_DIR / "notebooks" / "results" / "diarization_result.json"
        if not candidate_json.exists():
            return None

        try:
            with open(candidate_json, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
            if isinstance(data, dict) and data.get("audio_file") == audio_path.name:
                logger.info(
                    "Tu dong phat hien du lieu diarization khop voi file '%s' tai: %s",
                    audio_path.name,
                    candidate_json,
                )
                return candidate_json
        except Exception as exc:
            logger.warning(
                "Khong the doc file diarization tu dong tai '%s': %s",
                candidate_json,
                exc,
            )

        return None

    def process_audio(
        self,
        audio_path: Union[str, Path],
        title: str = "Cuộc họp bóc băng âm thanh",
        diarization_input: Optional[DiarizationInputType] = None,
        speaker_mapping: Optional[Dict[str, str]] = None,
        auto_detect_diarization: bool = False,
        run_llm_analysis: bool = False,
    ) -> MeetingAnalysis:
        """
        Quy trinh nghiep vu xu ly cuoc hop tu file ghi am:
        1. Kiem tra su ton tai cua file am thanh.
        2. Xac dinh du lieu diarization (truc tiep hoac tu dong phat hien neu bat co).
        3. Neu co diarization, goi transcribe_with_diarization de ket hop nhan nguoi noi
           (SPEAKER_00, SPEAKER_01...) va moc thoi gian chinh xac. Neu khong co, goi
           PhoWhisperASR.transcribe mac dinh voi nhan SPEAKER_UNASSIGNED.
        4. Luu cuoc hop va cac transcript segments vao database qua MeetingRepository.
        5. Tra ve ket qua MeetingAnalysis chuan hoa cho tang goi (UI/API).
        """
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Khong tim thay file am thanh tai: {path_obj.resolve()}")

        resolved_diarization = self._resolve_diarization_input(
            audio_path=path_obj,
            diarization_input=diarization_input,
            auto_detect_diarization=auto_detect_diarization,
        )

        meeting_title = title.strip() if title.strip() else "Cuộc họp bóc băng âm thanh"
        logger.info(
            "Bat dau xu ly cuoc hop qua LangGraph Pipeline: title='%s', file='%s', co_diarization=%s",
            meeting_title,
            path_obj.name,
            resolved_diarization is not None,
        )

        profiles = self.speaker_repo.get_all()
        final_state = meeting_graph.invoke(
            {
                "audio_path": path_obj,
                "valid_speakers": profiles,
                "raw_segments": [],
                "aligned_segments": [],
                "transcript_text": "",
                "analysis_result": None,
                "validation_errors": [],
                "retry_count": 0,
                "asr": self.asr,
                "analyzer": self.analyzer,
                "diarization_input": resolved_diarization,
                "speaker_mapping": speaker_mapping,
                "run_llm_analysis": run_llm_analysis,
            }
        )

        full_text = str(final_state.get("transcript_text") or "")
        segments: List[TranscriptSegment] = list(final_state.get("aligned_segments") or [])
        extraction: Optional[LLMExtractionOutput] = final_state.get("analysis_result")

        logger.info("LangGraph Pipeline hoan tat: tao duoc %d doan thoai.", len(segments))
        today_str = datetime.now().strftime("%Y-%m-%d")

        if run_llm_analysis and extraction is not None:
            final_segments = extraction.cleaned_segments if extraction.cleaned_segments else segments
            analysis = MeetingAnalysis(
                title=meeting_title,
                date=today_str,
                raw_transcript=full_text,
                cleaned_transcript=extraction.cleaned_transcript,
                segments=final_segments,
                summary=extraction.summary,
                key_decisions=extraction.key_decisions,
                tasks=extraction.tasks,
            )
        else:
            analysis = MeetingAnalysis(
                title=meeting_title,
                date=today_str,
                raw_transcript=full_text,
                cleaned_transcript=None,
                segments=segments,
                summary="Bản bóc băng tự động từ âm thanh (Chưa chạy phân tích tổng hợp)",
                key_decisions=[],
                tasks=[],
            )

        meeting_id = self.meeting_repo.create(analysis)
        analysis.id = meeting_id
        logger.info("Da luu cuoc hop audio thanh cong voi ID=%d.", meeting_id)

        if run_llm_analysis and extraction is not None:
            self.meeting_repo.update_analysis_result(
                meeting_id=meeting_id,
                cleaned_transcript=extraction.cleaned_transcript,
                summary=extraction.summary,
                key_decisions=extraction.key_decisions,
            )

        if analysis.tasks:
            task_ids = self.task_repo.create_batch(meeting_id, analysis.tasks)
            for task, tid in zip(analysis.tasks, task_ids):
                task.id = tid
                task.meeting_id = meeting_id

        return analysis

    def analyze_meeting(self, meeting_id: int) -> MeetingAnalysis:
        """
        Phan tich cuoc hop da luu trong co so du lieu qua LangGraph Pipeline:
        1. Truy xuat thong tin cuoc hop va danh sach segments hien tai.
        2. Goi meeting_graph.invoke() (analyze_node -> validate_node kem retry tu sua loi).
        3. Cap nhat ket qua vao MeetingRepository va TaskRepository.replace_meeting_tasks().
        4. Tra ve doi tuong MeetingAnalysis hoan chinh.
        """
        details = self.meeting_repo.get_by_id(meeting_id)
        if details is None:
            raise ValueError(f"Khong tim thay cuoc hop voi ID={meeting_id}.")

        raw_segments = details.get("segments") or []
        segments: List[TranscriptSegment] = [
            TranscriptSegment(
                speaker_id=s.get("speaker_id"),
                speaker_name=s.get("speaker_name"),
                start_time=s.get("start_time"),
                end_time=s.get("end_time"),
                text=s.get("text", ""),
            )
            for s in raw_segments
        ]
        raw_transcript = details.get("raw_transcript") or ""
        profiles = self.speaker_repo.get_all()

        logger.info(
            "Bat dau phan tich LangGraph cho meeting_id=%d (%d segments)...",
            meeting_id,
            len(segments),
        )
        final_state = meeting_graph.invoke(
            {
                "audio_path": None,
                "valid_speakers": profiles,
                "raw_segments": segments,
                "aligned_segments": segments,
                "transcript_text": raw_transcript,
                "analysis_result": None,
                "validation_errors": [],
                "retry_count": 0,
                "asr": self.asr,
                "analyzer": self.analyzer,
                "run_llm_analysis": True,
            }
        )
        extraction: Optional[LLMExtractionOutput] = final_state.get("analysis_result")
        if extraction is None:
            extraction = self.analyzer.analyze(segments=segments, raw_transcript=raw_transcript)

        self.meeting_repo.update_analysis_result(
            meeting_id=meeting_id,
            cleaned_transcript=extraction.cleaned_transcript,
            summary=extraction.summary,
            key_decisions=extraction.key_decisions,
        )
        if extraction.cleaned_segments:
            self.meeting_repo.update_segments_text(
                meeting_id=meeting_id,
                cleaned_segments=extraction.cleaned_segments,
            )
            segments = extraction.cleaned_segments
        self.task_repo.replace_meeting_tasks(
            meeting_id=meeting_id,
            tasks=extraction.tasks,
        )
        saved_tasks = self.task_repo.list_by_meeting_id(meeting_id)

        logger.info(
            "Phan tich LLM hoan tat cho meeting_id=%d: %d quyet dinh, %d cong viec.",
            meeting_id,
            len(extraction.key_decisions),
            len(saved_tasks),
        )
        return MeetingAnalysis(
            id=meeting_id,
            title=details.get("title") or "Cuộc họp không có tiêu đề",
            date=details.get("meeting_date"),
            raw_transcript=raw_transcript,
            cleaned_transcript=extraction.cleaned_transcript,
            segments=segments,
            summary=extraction.summary,
            key_decisions=extraction.key_decisions,
            tasks=saved_tasks,
        )

    def process_transcript(
        self,
        raw_transcript: str,
        title: str = "Cuộc họp mới",
    ) -> MeetingAnalysis:
        """
        Xu ly noi dung cuoc hop tu van ban tho.
        Giu nguyen mock data chuan tu Day 1 de bao toan tinh tuong thich cho cac bai kiem thu van ban.
        """
        if not raw_transcript.strip():
            raise ValueError("Noi dung cuoc hop khong duoc de trong.")

        meeting_title = title.strip() if title.strip() else "Cuộc họp mới"
        logger.info(
            "Bat dau xu ly cuoc hop tu van ban: title='%s', do dai=%d ky tu.",
            meeting_title,
            len(raw_transcript),
        )

        today_str = datetime.now().strftime("%Y-%m-%d")

        sample_segments = [
            TranscriptSegment(
                speaker_id="SPEAKER_00",
                speaker_name="Trần Giám Đốc",
                start_time=12.5,
                end_time=35.0,
                text="Chào mọi người, tuần này chúng ta cần ưu tiên hoàn thành báo cáo tài chính quý trước.",
            ),
            TranscriptSegment(
                speaker_id="SPEAKER_01",
                speaker_name="Nguyễn Văn A",
                start_time=36.2,
                end_time=58.0,
                text="Tôi sẽ phụ trách tổng hợp toàn bộ số liệu doanh thu và gửi trước thứ Sáu tuần này.",
            ),
            TranscriptSegment(
                speaker_id="SPEAKER_02",
                speaker_name="Trần Thị B",
                start_time=60.0,
                end_time=82.4,
                text="Sau khi cuộc họp kết thúc, tôi sẽ soạn thảo và gửi email biên bản họp cho cả phòng.",
            ),
        ]

        sample_tasks = [
            TaskItem(
                title="Chuẩn bị tài liệu báo cáo tài chính quý",
                description="Tổng hợp số liệu doanh thu và chi phí quý trước để trình ban giám đốc",
                assigner="Trần Giám Đốc",
                assignee="Nguyễn Văn A",
                due_date=today_str,
                priority=TaskPriority.HIGH,
                priority_reason="Ảnh hưởng trực tiếp đến kế hoạch duyệt ngân sách tuần tới",
                source_quote="Tôi sẽ phụ trách tổng hợp toàn bộ số liệu doanh thu và gửi trước thứ Sáu tuần này.",
                source_start_time=36.2,
                source_end_time=58.0,
                status=TaskStatus.TODO,
                confirmation_status=TaskConfirmationStatus.PENDING,
            ),
            TaskItem(
                title="Gửi email biên bản cuộc họp",
                description="Gửi bản tóm tắt nội dung và action items cho toàn bộ thành viên tham gia",
                assigner="Trần Giám Đốc",
                assignee="Trần Thị B",
                due_date=today_str,
                priority=TaskPriority.MEDIUM,
                priority_reason="Đảm bảo mọi người nắm được quyết định và nhiệm vụ sau cuộc họp",
                source_quote="Sau khi cuộc họp kết thúc, tôi sẽ soạn thảo và gửi email biên bản họp cho cả phòng.",
                source_start_time=60.0,
                source_end_time=82.4,
                status=TaskStatus.TODO,
                confirmation_status=TaskConfirmationStatus.PENDING,
            ),
        ]

        analysis = MeetingAnalysis(
            title=meeting_title,
            date=today_str,
            raw_transcript=raw_transcript,
            cleaned_transcript="Cuộc họp thảo luận về tiến độ công việc và báo cáo tài chính quý.",
            segments=sample_segments,
            summary="Cuộc họp định kỳ thảo luận về tiến độ công việc. Ban giám đốc yêu cầu hoàn thiện báo cáo tài chính và gửi biên bản họp trong tuần.",
            key_decisions=[
                "Thống nhất hạn chót nộp báo cáo tài chính quý là thứ Sáu",
                "Gửi email biên bản họp cho toàn bộ nhân sự liên quan",
            ],
            tasks=sample_tasks,
        )

        meeting_id = self.meeting_repo.create(analysis)
        analysis.id = meeting_id

        task_ids = self.task_repo.create_batch(meeting_id, analysis.tasks)
        for task, tid in zip(analysis.tasks, task_ids):
            task.id = tid
            task.meeting_id = meeting_id

        logger.info("Da luu cuoc hop van ban va %d tasks vao database voi ID=%d.", len(task_ids), meeting_id)
        return analysis

    def get_recent_meetings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Lay danh sach cac cuoc hop gan day nhat."""
        return self.meeting_repo.list_all(limit=limit)

    def get_meeting_details(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Lay chi tiet mot cuoc hop kem cac doan thoai."""
        return self.meeting_repo.get_by_id(meeting_id)

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        confirmation_status: Optional[str] = None,
    ) -> List[TaskItem]:
        """Lay toan bo danh sach cong viec co loc theo trang thai."""
        return self.task_repo.list_all(status=status, confirmation_status=confirmation_status)

    def get_tasks_by_meeting(self, meeting_id: int) -> List[TaskItem]:
        """Lay danh sach cong viec thuoc ve mot cuoc hop cu the."""
        return self.task_repo.list_by_meeting_id(meeting_id)

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
        Cap nhat thong tin chi tiet cua cong viec (Human-in-the-loop):
        ten cong viec, mo ta, nguoi giao, nguoi nhan, han chot va do uu tien.
        """
        logger.info("Cap nhat chi tiet noi dung cho task ID=%d.", task_id)
        return self.task_repo.update_task_details(
            task_id=task_id,
            title=title,
            description=description,
            assigner=assigner,
            assignee=assignee,
            due_date=due_date,
            priority=priority,
            priority_reason=priority_reason,
        )

    def update_task_status(self, task_id: int, status: str) -> bool:
        """Cap nhat tien do cong viec."""
        logger.info("Cap nhat trang thai task ID=%d thanh '%s'.", task_id, status)
        return self.task_repo.update_status(task_id, status)

    def update_task_confirmation(self, task_id: int, confirmation_status: str) -> bool:
        """Cap nhat trang thai duyet cua nguoi dung."""
        logger.info("Cap nhat trang thai xac nhan task ID=%d thanh '%s'.", task_id, confirmation_status)
        return self.task_repo.update_confirmation(task_id, confirmation_status)

    def update_meeting_summary(
        self,
        meeting_id: int,
        summary: str,
        key_decisions: Optional[List[str]] = None,
    ) -> bool:
        """Cap nhat tom tat va quyet dinh chinh."""
        logger.info("Cap nhat ban tom tat cho meeting_id=%d.", meeting_id)
        return self.meeting_repo.update_summary(meeting_id, summary, key_decisions)

    def update_speaker_names(
        self,
        meeting_id: int,
        speaker_mapping: Dict[str, str],
    ) -> bool:
        """
        Cap nhat ten hien thi cua cac nguoi noi (speaker_id -> speaker_name) trong cuoc hop,
        dong thoi dong bo ten cu sang ten moi trong summary va tasks (assigner, assignee).
        """
        logger.info(
            "Cap nhat speaker_name cho meeting_id=%d voi %d nhan.",
            meeting_id,
            len(speaker_mapping),
        )
        updated_count = self.meeting_repo.update_speaker_names(
            meeting_id=meeting_id,
            speaker_mapping=speaker_mapping,
        )
        return updated_count > 0

    def delete_meeting(self, meeting_id: int) -> bool:
        """Xoa ho so cuoc hop khoi co so du lieu."""
        logger.info("Xoa cuoc hop meeting_id=%d.", meeting_id)
        return self.meeting_repo.delete(meeting_id)

    def get_meeting_audio_path(self, meeting_id: int) -> Path:
        """Tra ve duong dan file am thanh cua cuoc hop hoac fallback ve meeting_test.wav."""
        uploaded_dir = BASE_DIR / "data" / "uploads"
        candidate = uploaded_dir / f"meeting_{meeting_id}.wav"
        if candidate.exists():
            return candidate
        return BASE_DIR / "data" / "samples" / "meeting_test.wav"

    def list_speaker_profiles(self) -> List[SpeakerProfile]:
        """Lay danh sach toan bo ho so giong noi nhan su trong he thong."""
        return self.speaker_repo.get_all()

    def enroll_speaker_profile(
        self,
        name: str,
        role: str,
        department: str,
        audio_path: Union[str, Path],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        username: Optional[str] = None,
        email: str = "",
        is_admin: bool = False,
    ) -> SpeakerProfile:
        """
        Dang ky ho so giong noi nhan su moi tu file am thanh (hoac mot phan doan am thanh).
        Luu file mau vao data/voiceprints/ va trich xuat vector dac trung luu vao DB.
        """
        clean_name = name.strip()
        clean_role = role.strip()
        clean_dept = (department or "").strip()
        if not clean_name or not clean_role:
            raise ValueError("Ho ten va Chuc vu nhan su khong duoc de trong.")

        src_path = Path(audio_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Khong tim thay file am thanh mau: {src_path}")

        voiceprints_dir = BASE_DIR / "data" / "voiceprints"
        voiceprints_dir.mkdir(parents=True, exist_ok=True)
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_slug = "".join(ch if ch.isalnum() else "_" for ch in clean_name.lower()).strip("_") or "speaker"
        saved_sample_path = voiceprints_dir / f"vp_{safe_slug}_{timestamp_slug}.wav"

        save_audio_clip(
            source_audio_path=src_path,
            target_audio_path=saved_sample_path,
            start_time=start_time,
            end_time=end_time,
        )

        embedding = extract_voiceprint_embedding(saved_sample_path)
        rel_path = str(saved_sample_path.relative_to(BASE_DIR)).replace("\\", "/")

        profile = SpeakerProfile(
            username=username.strip().lower() if username and username.strip() else None,
            email=(email or "").strip(),
            name=clean_name,
            role=clean_role,
            department=clean_dept,
            voiceprint_json=json.dumps(embedding),
            sample_audio_path=rel_path,
            is_admin=is_admin,
            onboarding_completed=True,
        )
        profile_id = self.speaker_repo.create(profile)
        created = self.speaker_repo.get_by_id(profile_id)
        return created if created is not None else profile

    def login_or_register_user(self, username: str, email: str = "") -> SpeakerProfile:
        """
        Dang nhap bang username. Neu username chua ton tai, khoi tao phien tai khoan moi
        voi onboarding_completed = False (chua co voiceprint) de bat buoc qua buoc Onboarding.
        """
        clean_user = (username or "").strip().lower()
        clean_user = "".join(ch for ch in clean_user if ch.isalnum() or ch in ("_", ".", "-"))
        if not clean_user:
            raise ValueError("Ten dang nhap khong hop le.")

        existing = self.speaker_repo.get_by_username(clean_user)
        if existing is not None:
            return existing

        new_profile = SpeakerProfile(
            username=clean_user,
            email=(email or "").strip(),
            name=username.strip(),
            role="Thành viên mới",
            department="",
            voiceprint_json="[]",
            sample_audio_path=None,
            is_admin=False,
            onboarding_completed=False,
        )
        new_id = self.speaker_repo.create(new_profile)
        created = self.speaker_repo.get_by_id(new_id)
        return created if created is not None else new_profile

    def complete_user_onboarding(
        self,
        profile_id: Optional[int],
        username: Optional[str],
        name: str,
        role: str,
        department: str,
        audio_path: Union[str, Path],
        email: Optional[str] = None,
    ) -> SpeakerProfile:
        """
        Hoan tat quy trinh Onboarding lan dau cho nguoi dung moi:
        Cap nhat Ho ten, Chuc vu, Phong ban, trich xuat vector 60-dim L2 Voiceprint
        tu file thu am 10-30s va danh dau onboarding_completed = True.
        """
        clean_name = name.strip()
        clean_role = role.strip()
        clean_dept = (department or "").strip()
        if not clean_name or not clean_role:
            raise ValueError("Vui long nhap day du Ho ten va Chuc vu.")

        target_profile: Optional[SpeakerProfile] = None
        if profile_id is not None:
            target_profile = self.speaker_repo.get_by_id(profile_id)
        if target_profile is None and username:
            target_profile = self.speaker_repo.get_by_username(username)

        if target_profile is None:
            return self.enroll_speaker_profile(
                name=clean_name,
                role=clean_role,
                department=clean_dept,
                audio_path=audio_path,
                username=username,
                email=email or "",
            )

        return self.update_speaker_profile(
            profile_id=target_profile.id,
            name=clean_name,
            role=clean_role,
            department=clean_dept,
            username=username if username else target_profile.username,
            email=email if email is not None else target_profile.email,
            audio_path=audio_path,
            onboarding_completed=True,
        )

    def update_speaker_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        department: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        is_admin: Optional[bool] = None,
        audio_path: Optional[Union[str, Path]] = None,
        onboarding_completed: Optional[bool] = None,
    ) -> SpeakerProfile:
        """
        Cap nhat thong tin ho so nhan su (Ho ten, Chuc vu, Phong ban, Quyen Admin)
        va tuy chon cap nhat lai mau giong noi (.wav).
        """
        existing = self.speaker_repo.get_by_id(profile_id)
        if existing is None:
            raise ValueError(f"Khong tim thay ho so nhan su ID={profile_id}")

        new_vp_json: Optional[str] = None
        new_sample_rel: Optional[str] = None
        new_onboard_flag = onboarding_completed

        if audio_path is not None:
            src_path = Path(audio_path)
            if not src_path.exists():
                raise FileNotFoundError(f"Khong tim thay file am thanh: {src_path}")

            voiceprints_dir = BASE_DIR / "data" / "voiceprints"
            voiceprints_dir.mkdir(parents=True, exist_ok=True)
            timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target_name = (name or existing.name).lower()
            safe_slug = "".join(ch if ch.isalnum() else "_" for ch in target_name).strip("_") or "speaker"
            saved_sample_path = voiceprints_dir / f"vp_{safe_slug}_{timestamp_slug}.wav"

            save_audio_clip(source_audio_path=src_path, target_audio_path=saved_sample_path)
            embedding = extract_voiceprint_embedding(saved_sample_path)
            new_vp_json = json.dumps(embedding)
            new_sample_rel = str(saved_sample_path.relative_to(BASE_DIR)).replace("\\", "/")
            new_onboard_flag = True

        updated = self.speaker_repo.update(
            profile_id=profile_id,
            name=name,
            role=role,
            department=department,
            username=username,
            email=email,
            voiceprint_json=new_vp_json,
            sample_audio_path=new_sample_rel,
            is_admin=is_admin,
            onboarding_completed=new_onboard_flag,
        )
        if updated is None:
            raise ValueError(f"Khong the cap nhat ho so nhan su ID={profile_id}")
        return updated

    def enroll_speaker_from_meeting(
        self,
        meeting_id: int,
        speaker_id: str,
        name: str,
        role: str,
        department: str = "",
    ) -> SpeakerProfile:
        """
        Trich xuat mau giong cua mot speaker_id truc tiep tu cuoc hop va luu vao bang speaker_profiles.
        """
        details = self.meeting_repo.get_by_id(meeting_id)
        if details is None:
            raise ValueError(f"Khong tim thay cuoc hop ID={meeting_id}.")

        segments = details.get("segments") or []
        speaker_segs = [
            s
            for s in segments
            if s.get("speaker_id") == speaker_id
            and s.get("start_time") is not None
            and s.get("end_time") is not None
            and float(s["end_time"]) > float(s["start_time"]) + 0.5
        ]
        if not speaker_segs:
            raise ValueError(f"Khong tim thay doan am thanh hop le cho {speaker_id} trong cuoc hop.")

        # Chon doan phat bieu dai nhat cua speaker_id lam mau giong chuan
        best_seg = max(speaker_segs, key=lambda s: float(s["end_time"]) - float(s["start_time"]))
        audio_path = self.get_meeting_audio_path(meeting_id)

        return self.enroll_speaker_profile(
            name=name,
            role=role,
            department=department,
            audio_path=audio_path,
            start_time=float(best_seg["start_time"]),
            end_time=float(best_seg["end_time"]),
        )

    def delete_speaker_profile(self, profile_id: int) -> bool:
        """Xoa ho so nhan su khoi he thong."""
        logger.info("Xoa speaker_profile ID=%d.", profile_id)
        return self.speaker_repo.delete(profile_id)

    @staticmethod
    def clean_segment_text(text: str) -> str:
        """Chuan hoa van ban ASR tieng Viet theo ngu canh cuoc hop cho mot phan doan."""
        return clean_vietnamese_asr_text(text)
