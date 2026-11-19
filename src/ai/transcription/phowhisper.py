"""
Module: phowhisper.py
Duong dan: src/ai/transcription/phowhisper.py
Muc dich: Module ASR (Automatic Speech Recognition) chinh thuc cua he thong.
Su dung mo hinh PhoWhisper (vinai/PhoWhisper-base) de nhan dang giong noi tieng Viet.

Quy tac kien truc va do chinh xac du lieu:
- Khong import Streamlit hoac bat ky thu vien UI nao.
- Ap dung Lazy Loading: chi nap mo hinh khi thuc su can su dung.
- Tai su dung pipeline (in-memory caching): khong nap lai mo hinh o cac lan goi sau.
- Dong bo hoa thread-safe: su dung khoa reentrant/lock tranh tranh chap khi nhieu phien goi dong thoi.
- Trung thuc ve timestamp: bao toan gia tri None khi mo hinh khong du doan duoc end_time.
- Chuyen doi ket qua sang danh sach Pydantic TranscriptSegment chuan hoa.
- Doc file WAV truc tiep bang SoundFile, khong phu thuoc vao FFmpeg he thong.
"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from src.schemas.meeting import TranscriptSegment

logger = logging.getLogger(__name__)


class PhoWhisperASR:
    """
    Dieu phoi nhan dang giong noi tieng Viet su dung mo hinh PhoWhisper.
    Ho tro trich xuat moc thoi gian (timestamps) va dong goi TranscriptSegment.
    
    Luu y: Lop nay chi thuc hien ASR (Speech-to-Text), chua thuc hien Speaker Diarization.
    Cac gia tri speaker_id va speaker_name la placeholder cho den khi tich hop Diarization.
    """

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-base",
        device: str = "cpu",
        chunk_length_s: int = 30,
    ) -> None:
        """
        Khoi tao cau hinh cho PhoWhisperASR.
        Ap dung Lazy Loading: khong nap pipeline vao RAM ngay tai thoi diem khoi tao.
        """
        self.model_id = model_id
        self.device = device
        self.chunk_length_s = chunk_length_s
        self._pipeline = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Kiem tra xem mo hinh da duoc nap vao bo nho hay chua."""
        return self._pipeline is not None

    def load_model(self) -> None:
        """
        Nap mo hinh PhoWhisper vao bo nho su dung double-checked locking de dam bao thread-safe.
        Neu mo hinh da ton tai, bo qua de tiet kiem tai nguyen.
        """
        if self._pipeline is not None:
            return

        with self._lock:
            if self._pipeline is not None:
                return

            from transformers import pipeline

            logger.info(
                "Dang nap mo hinh PhoWhisper '%s' len thiet bi '%s'...",
                self.model_id,
                self.device,
            )
            start_time = time.time()

            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                chunk_length_s=self.chunk_length_s,
            )

            load_duration = time.time() - start_time
            logger.info(
                "Nap mo hinh PhoWhisper thanh cong trong %.2f giay.",
                load_duration,
            )

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        return_timestamps: bool = True,
        speaker_id: str = "SPEAKER_00",
        speaker_name: str = "Nguoi noi",
    ) -> Tuple[str, List[TranscriptSegment]]:
        """
        Boc bang am thanh thanh van ban tieng Viet kem moc thoi gian.

        Tham so:
        - audio_input: Duong dan file am thanh (str, Path) hoac mang numpy audio data.
        - return_timestamps: True de trich xuat moc thoi gian theo tung doan loi thoai.
        - speaker_id: Nhan dinh danh nguoi noi (placeholder).
        - speaker_name: Ten hien thi nguoi noi (placeholder).

        Tra ve:
        - Tuple[str, List[TranscriptSegment]]: Full text va danh sach cac doan hoi thoai.
        """
        self.load_model()

        audio_data = None
        sample_rate = 16000
        total_duration = None

        # 1. Doc va kiem tra tinh hop le cua du lieu am thanh
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Khong tim thay file am thanh tai: {audio_path.resolve()}")

            try:
                audio_data, sample_rate = sf.read(str(audio_path))
            except Exception as e:
                raise ValueError(
                    f"Khong the doc file am thanh '{audio_path}'. Dinh dang file khong duoc ho tro. Chi tiet: {e}"
                ) from e

            if len(audio_data) == 0:
                raise ValueError(f"File am thanh '{audio_path}' rong (0 samples).")

            total_duration = len(audio_data) / float(sample_rate)

            # Chuyen stereo sang mono neu can thiet
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            inputs = {"raw": audio_data, "sampling_rate": sample_rate}
        elif isinstance(audio_input, np.ndarray):
            audio_data = audio_input
            if len(audio_data) == 0:
                raise ValueError("Mang du lieu am thanh audio_input rong (0 samples).")

            total_duration = len(audio_data) / float(sample_rate)
            inputs = {"raw": audio_data, "sampling_rate": sample_rate}
        else:
            raise ValueError(
                f"Dinh dang audio_input khong hop le: {type(audio_input)}. Chi chap nhan str, Path hoac np.ndarray."
            )

        # 2. Thuc hien suy luan co khoa lock de an toan khi chia se instance tren nhieu luong
        logger.info("Bat dau giai ma am thanh va trich xuat van ban...")
        start_infer = time.time()

        with self._lock:
            result = self._pipeline(
                inputs,
                return_timestamps=return_timestamps,
            )

        infer_duration = time.time() - start_infer
        logger.info("Boc bang am thanh hoan tat trong %.2f giay.", infer_duration)

        full_text = result.get("text", "").strip()

        # 3. Chuyen doi chunks sang TranscriptSegment (giu nguyen None neu model khong doan duoc moc ket thuc)
        segments: List[TranscriptSegment] = []
        raw_chunks = result.get("chunks", [])

        if raw_chunks:
            for chunk in raw_chunks:
                chunk_text = chunk.get("text", "").strip()
                if not chunk_text:
                    continue

                ts = chunk.get("timestamp")
                start_sec = None
                end_sec = None

                if isinstance(ts, (tuple, list)):
                    if len(ts) > 0 and ts[0] is not None:
                        start_sec = round(float(ts[0]), 2)

                    if len(ts) > 1 and ts[1] is not None:
                        raw_end = float(ts[1])
                        if total_duration is not None and raw_end > total_duration:
                            end_sec = round(total_duration, 2)
                        else:
                            end_sec = round(raw_end, 2)

                segments.append(
                    TranscriptSegment(
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        start_time=start_sec,
                        end_time=end_sec,
                        text=chunk_text,
                    )
                )

        if not segments and full_text:
            segments.append(
                TranscriptSegment(
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    start_time=None,
                    end_time=None,
                    text=full_text,
                )
            )

        return full_text, segments
