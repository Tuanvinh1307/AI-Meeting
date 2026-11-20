"""
Module: phowhisper.py
Duong dan: src/core/ai/asr/phowhisper.py
Muc dich: Module ASR (Automatic Speech Recognition) chinh thuc cua he thong.
Su dung mo hinh PhoWhisper (vinai/PhoWhisper-base) de nhan dang giong noi tieng Viet.

Quy tac kien truc va do chinh xac du lieu:
- Khong import Streamlit hoac bat ky thu vien UI nao.
- Ap dung Lazy Loading: chi nap mo hinh khi thuc su can su dung.
- Tai su dung pipeline (in-memory caching): khong nap lai mo hinh o cac lan goi sau.
- Dong bo hoa thread-safe: su dung khoa lock tranh tranh chap khi nhieu phien goi dong thoi.
- Su dung co che native timestamp cua PhoWhisper/Whisper (khong tu cat audio theo khoang lang
  truoc khi boc bang de tranh cat dut giua tu).
- Neu Whisper tra ve end_time = None, lay thoi luong thuc te cua file audio (tinh bang soundfile)
  lam end_time chinh xac.
"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from src.schemas.meeting import TranscriptSegment

logger = logging.getLogger(__name__)


def _resolve_default_device(device: Optional[str] = None) -> str:
    """Tu dong chon GPU ('cuda') neu kha dung, nguoc lai dung 'cpu'."""
    if device is not None and str(device).strip():
        req = str(device).strip().lower()
        if req.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_audio_by_silence(
    audio_data: np.ndarray,
    sample_rate: int,
    min_segment_s: float = 3.0,
    max_segment_s: float = 14.0,
    min_silence_s: float = 0.28,
    padding_s: float = 0.12,
) -> List[Tuple[float, float]]:
    """Ham giu lai de tuong thich import; khong su dung de cat audio truoc khi chay PhoWhisper."""
    if len(audio_data) == 0 or sample_rate <= 0:
        return []
    return [(0.0, round(len(audio_data) / float(sample_rate), 2))]


class PhoWhisperASR:
    """
    Dieu phoi nhan dang giong noi tieng Viet su dung mo hinh PhoWhisper.
    Su dung truc tiep native timestamp cua Whisper va lay thoi luong thuc cua audio
    khi Whisper tra ve ending timestamp = None.
    """

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-base",
        device: Optional[str] = None,
        chunk_length_s: int = 30,
    ) -> None:
        self.model_id = model_id
        self.device = _resolve_default_device(device)
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
        Khi chay tren GPU ('cuda'), truyen dung thiet bi GPU vao transformers.pipeline.
        """
        if self._pipeline is not None:
            return

        with self._lock:
            if self._pipeline is not None:
                return

            from transformers import pipeline

            pipeline_device: Union[int, str, torch.device] = (
                0 if self.device == "cuda" and torch.cuda.is_available() else self.device
            )

            logger.info(
                "Dang nap mo hinh PhoWhisper '%s' len thiet bi '%s' (cuda_available=%s)...",
                self.model_id,
                self.device,
                torch.cuda.is_available(),
            )
            start_time = time.time()

            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
                device=pipeline_device,
            )

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            load_duration = time.time() - start_time
            logger.info(
                "Nap mo hinh PhoWhisper thanh cong tren '%s' trong %.2f giay.",
                self.device,
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
        Boc bang am thanh thanh van ban tieng Viet bang co che native cua PhoWhisper:
        - Khong tu cat audio bang VAD truoc khi dua vao PhoWhisper de bao toan 100% ngu canh am vi.
        - Neu Whisper tra ve end_time = None o chunk cuoi, gan end_time bang thoi luong thuc te
          cua file audio (total_duration tinh tu soundfile).
        """
        self.load_model()

        sample_rate = 16000
        total_duration: Optional[float] = None

        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Khong tim thay file am thanh tai: {audio_path.resolve()}"
                )

            try:
                audio_data, sample_rate = sf.read(str(audio_path))
            except Exception as exc:
                raise ValueError(
                    f"Khong the doc file am thanh '{audio_path}'. Dinh dang file khong duoc ho tro. Chi tiet: {exc}"
                ) from exc

            if len(audio_data) == 0:
                raise ValueError(f"File am thanh '{audio_path}' rong (0 samples).")

            total_duration = len(audio_data) / float(sample_rate)

            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            inputs = {"raw": audio_data, "sampling_rate": sample_rate}
        elif isinstance(audio_input, np.ndarray):
            audio_data = audio_input
            if len(audio_data) == 0:
                raise ValueError("Mang du lieu am thanh audio_input rong (0 samples).")

            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            total_duration = len(audio_data) / float(sample_rate)
            inputs = {"raw": audio_data, "sampling_rate": sample_rate}
        else:
            raise ValueError(
                f"Dinh dang audio_input khong hop le: {type(audio_input)}. Chi chap nhan str, Path hoac np.ndarray."
            )

        logger.info("Bat dau giai ma am thanh va trich xuat van ban...")
        start_infer = time.time()

        with self._lock:
            try:
                if return_timestamps:
                    result = self._pipeline(inputs, return_timestamps=True)
                else:
                    result = self._pipeline(inputs)
            finally:
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        infer_duration = time.time() - start_infer
        logger.info("Boc bang am thanh hoan tat trong %.2f giay.", infer_duration)

        full_text = result.get("text", "").strip() if isinstance(result, dict) else str(result or "").strip()

        segments: List[TranscriptSegment] = []
        raw_chunks = result.get("chunks", []) if isinstance(result, dict) else []

        if raw_chunks:
            num_chunks = len(raw_chunks)
            for idx, chunk in enumerate(raw_chunks):
                chunk_text = chunk.get("text", "").strip()
                if not chunk_text:
                    continue

                ts = chunk.get("timestamp")
                start_sec: Optional[float] = None
                end_sec: Optional[float] = None

                if isinstance(ts, (tuple, list)):
                    if len(ts) > 0 and ts[0] is not None:
                        start_sec = round(float(ts[0]), 2)

                    if len(ts) > 1 and ts[1] is not None:
                        raw_end = float(ts[1])
                        if total_duration is not None and raw_end > total_duration:
                            end_sec = round(total_duration, 2)
                        else:
                            end_sec = round(raw_end, 2)

                # Neu Whisper tra ve end_time = None, lay moc bat dau cua chunk ke tiep
                # hoac lay thoi luong thuc te cua file audio (total_duration)
                if end_sec is None:
                    next_start: Optional[float] = None
                    if idx + 1 < num_chunks:
                        next_ts = raw_chunks[idx + 1].get("timestamp")
                        if (
                            isinstance(next_ts, (tuple, list))
                            and len(next_ts) > 0
                            and next_ts[0] is not None
                        ):
                            cand_next = float(next_ts[0])
                            if start_sec is None or cand_next > start_sec:
                                next_start = round(cand_next, 2)

                    if next_start is not None:
                        end_sec = next_start
                    elif total_duration is not None:
                        end_sec = round(total_duration, 2)

                segments.append(
                    TranscriptSegment(
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        start_time=start_sec if start_sec is not None else 0.0,
                        end_time=end_sec,
                        text=chunk_text,
                    )
                )

        if not segments and full_text:
            segments.append(
                TranscriptSegment(
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    start_time=0.0 if total_duration is not None else None,
                    end_time=round(total_duration, 2) if total_duration is not None else None,
                    text=full_text,
                )
            )

        return full_text, segments
