"""
Module: phowhisper.py
Duong dan: src/core/ai/asr/phowhisper.py
Muc dich: Module ASR (Automatic Speech Recognition) chinh thuc cua he thong.
Su dung mo hinh PhoWhisper (vinai/PhoWhisper-base) chay tren engine faster-whisper (CTranslate2)
voi luong hoa int8 (DP4A tren GTX 1060 sm_61) + Silero VAD.

Quy tac kien truc va do chinh xac du lieu:
- Khong import Streamlit hoac bat ky thu vien UI nao.
- Ap dung Lazy Loading & Singleton Caching: chi nap mo hinh 1 lan duy nhat vao VRAM.
- Tu dong kiem tra va convert model VinAI PhoWhisper sang dinh dang CTranslate2 (int8)
  luu trong volume cache ben vung (/root/.cache/huggingface/ct2_models) de tai su dung.
- Bat Silero VAD (vad_filter=True) va giu nguyen cau truc dau ra Tuple[str, List[TranscriptSegment]].
"""

import ctypes
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from src.schemas.meeting import TranscriptSegment

logger = logging.getLogger(__name__)

# Singleton cache chia se instance WhisperModel giua cac lan khoi tao PhoWhisperASR
_SHARED_WHISPER_MODELS: Dict[Tuple[str, str, str], Any] = {}
_SHARED_CACHE_LOCK = threading.Lock()
_CUDNN_PRELOADED = False


def _preload_nvidia_shared_libs() -> None:
    """
    Nap truoc cac thu vien chia se CUDA 12 / cuDNN 9 đi kem PyTorch (libcudnn_ops.so.9, libcublas.so.12...)
    de CTranslate2 / faster-whisper tim thay ngay tren Linux container ma khong bi loi missing shared object.
    """
    global _CUDNN_PRELOADED
    if _CUDNN_PRELOADED:
        return
    _CUDNN_PRELOADED = True

    if os.name == "nt":
        return

    try:
        import nvidia  # type: ignore[import-not-found]

        nvidia_root = Path(nvidia.__file__).resolve().parent
    except Exception:
        nvidia_root = Path("/usr/local/lib/python3.10/site-packages/nvidia")

    if not nvidia_root.exists():
        return

    lib_patterns = (
        "cuda_runtime/lib/libcudart.so*",
        "cublas/lib/libcublasLt.so*",
        "cublas/lib/libcublas.so*",
        "cudnn/lib/libcudnn.so*",
        "cudnn/lib/libcudnn_ops*.so*",
        "cudnn/lib/libcudnn_cnn*.so*",
        "cudnn/lib/libcudnn_adv*.so*",
    )
    for pattern in lib_patterns:
        for so_path in sorted(nvidia_root.glob(pattern)):
            if so_path.is_file():
                try:
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _resolve_default_device(device: Optional[str] = None) -> str:
    """Tu dong chon GPU ('cuda') neu kha dung, nguoc lai dung 'cpu'."""
    if device is not None and str(device).strip():
        req = str(device).strip().lower()
        if req.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return "cuda" if req.startswith("cuda") else req
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_ct2_cache_dir(model_id: str) -> Path:
    """
    Xac dinh thu muc luu model PhoWhisper da convert sang CTranslate2
    nam trong volume HF_HOME (/root/.cache/huggingface) de ben vung qua cac lan restart container.
    """
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        base_cache = Path(hf_home)
    else:
        base_cache = Path.home() / ".cache" / "huggingface"

    safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "__" for ch in model_id)
    return base_cache / "ct2_models" / f"{safe_slug}_ct2_int8"


def _ensure_ct2_phowhisper_model(model_id: str) -> Path:
    """
    Kiem tra model VinAI PhoWhisper da duoc convert sang CTranslate2 trong cache chua.
    Neu chua co, tu dong convert 1 lan duy nhat va sinh tokenizer.json chuan cho faster-whisper.
    """
    clean_id = model_id.strip()
    local_dir = Path(clean_id)
    if local_dir.exists() and (local_dir / "model.bin").exists():
        return local_dir

    target_dir = _get_ct2_cache_dir(clean_id)
    model_bin = target_dir / "model.bin"
    tokenizer_json = target_dir / "tokenizer.json"

    if model_bin.exists() and tokenizer_json.exists():
        logger.info("Tai su dung model PhoWhisper CTranslate2 trong cache: %s", target_dir)
        return target_dir

    logger.info(
        "Chua co model CTranslate2 tai '%s'. Bat dau chuyen doi 1 lan tu '%s' (quantization=int8)...",
        target_dir,
        clean_id,
    )
    start_t = time.time()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    from ctranslate2.converters import TransformersConverter
    from transformers import AutoProcessor, AutoTokenizer

    converter = TransformersConverter(model_name_or_path=clean_id)
    converter.convert(
        output_dir=str(target_dir),
        quantization="int8",
        force=True,
    )

    # Luu fast tokenizer.json va preprocessor_config.json de faster-whisper nap truc tiep
    tokenizer = AutoTokenizer.from_pretrained(clean_id)
    tokenizer.save_pretrained(str(target_dir))
    try:
        processor = AutoProcessor.from_pretrained(clean_id)
        processor.save_pretrained(str(target_dir))
    except Exception as exc:
        logger.debug("Bo qua luu AutoProcessor phu: %s", exc)

    logger.info(
        "Chuyen doi '%s' sang CTranslate2 (int8) hoan tat trong %.2fs tai: %s",
        clean_id,
        time.time() - start_t,
        target_dir,
    )
    return target_dir


def _resample_to_16khz(audio_data: np.ndarray, orig_sr: int) -> np.ndarray:
    """Chuan hoa tin hieu am thanh ve 16000 Hz mono float32 cho faster-whisper bang numpy."""
    if orig_sr == 16000 or len(audio_data) == 0 or orig_sr <= 0:
        return audio_data
    duration = len(audio_data) / float(orig_sr)
    target_len = max(1, int(round(duration * 16000)))
    x_old = np.linspace(0.0, duration, num=len(audio_data), endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, audio_data)
    return np.asarray(resampled, dtype=np.float32)


def split_audio_by_silence(
    audio_data: np.ndarray,
    sample_rate: int,
    min_segment_s: float = 3.0,
    max_segment_s: float = 14.0,
    min_silence_s: float = 0.28,
    padding_s: float = 0.12,
) -> List[Tuple[float, float]]:
    """Ham giu lai de tuong thich import; VAD nay do Silero VAD cua faster-whisper dam nhiem."""
    if len(audio_data) == 0 or sample_rate <= 0:
        return []
    return [(0.0, round(len(audio_data) / float(sample_rate), 2))]


class PhoWhisperASR:
    """
    Dieu phoi nhan dang giong noi tieng Viet su dung mo hinh PhoWhisper tren engine faster-whisper:
    - Chay compute_type='int8' toi uu cho GPU GTX 1060 (Pascal sm_61 DP4A) va CPU fallback.
    - Tich hop Silero VAD (vad_filter=True, min_silence_duration_ms=500).
    - Singleton caching: chi nap model vao VRAM 1 lan duy nhat.
    """

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-base",
        device: Optional[str] = None,
        chunk_length_s: int = 30,
        compute_type: str = "int8",
        beam_size: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
    ) -> None:
        self.model_id = os.getenv("ASR_MODEL_ID", model_id).strip() or "vinai/PhoWhisper-base"
        self.device = _resolve_default_device(device)
        self.chunk_length_s = chunk_length_s
        self.compute_type = os.getenv("ASR_COMPUTE_TYPE", compute_type).strip() or "int8"
        self.beam_size = (
            beam_size
            if beam_size is not None
            else int(os.getenv("ASR_BEAM_SIZE", "2"))
        )
        self.min_silence_duration_ms = (
            min_silence_duration_ms
            if min_silence_duration_ms is not None
            else int(os.getenv("ASR_VAD_MIN_SILENCE_MS", "500"))
        )
        self._pipeline: Any = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Kiem tra xem mo hinh da duoc nap vao bo nho hay chua."""
        return self._pipeline is not None

    def load_model(self) -> None:
        """
        Nap mo hinh PhoWhisper bang faster-whisper (WhisperModel) voi Singleton Cache.
        Neu chay tren GPU gap loi thieu thu vien CUDA/cuDNN, tu dong fallback sang CPU int8.
        """
        if self._pipeline is not None:
            return

        with self._lock:
            if self._pipeline is not None:
                return

            cache_key = (self.model_id, self.device, self.compute_type)
            with _SHARED_CACHE_LOCK:
                if cache_key in _SHARED_WHISPER_MODELS:
                    self._pipeline = _SHARED_WHISPER_MODELS[cache_key]
                    return

                _preload_nvidia_shared_libs()
                start_time = time.time()

                try:
                    from faster_whisper import WhisperModel

                    ct2_dir = _ensure_ct2_phowhisper_model(self.model_id)
                    logger.info(
                        "Dang nap faster-whisper PhoWhisper '%s' (dir='%s') len device='%s', compute_type='%s'...",
                        self.model_id,
                        ct2_dir,
                        self.device,
                        self.compute_type,
                    )
                    try:
                        model = WhisperModel(
                            str(ct2_dir),
                            device=self.device,
                            compute_type=self.compute_type,
                        )
                    except Exception as gpu_exc:
                        err_msg = (
                            f"[ASR] Khoi tao faster-whisper tren device='{self.device}' "
                            f"({self.compute_type}) that bai: {gpu_exc}"
                        )
                        print(err_msg, flush=True)
                        logger.warning(err_msg)
                        if self.device != "cpu":
                            print(
                                f"[ASR] Fallback faster-whisper sang device='cpu', compute_type='int8' (nguyen nhan GPU: {gpu_exc})",
                                flush=True,
                            )
                            self.device = "cpu"
                            self.compute_type = "int8"
                            cache_key = (self.model_id, self.device, self.compute_type)
                            model = WhisperModel(
                                str(ct2_dir),
                                device="cpu",
                                compute_type="int8",
                            )
                        else:
                            raise

                    _SHARED_WHISPER_MODELS[cache_key] = model
                    self._pipeline = model
                    loaded_msg = f"[ASR] Loaded faster-whisper on {self.device} ({self.compute_type})"
                    print(loaded_msg, flush=True)
                    logger.info(
                        "%s trong %.2f giay.",
                        loaded_msg,
                        time.time() - start_time,
                    )
                except Exception as exc:
                    # In day du noi dung loi import/khoi tao faster-whisper ra stdout kem flush=True
                    fail_msg = (
                        f"[ASR] Import hoac khoi tao faster-whisper that bai ({type(exc).__name__}: {exc}). "
                        f"Fallback ve transformers.pipeline tren '{self.device}'."
                    )
                    print(fail_msg, flush=True)
                    logger.warning(fail_msg)
                    from transformers import pipeline

                    pipeline_device: Union[int, str, torch.device] = (
                        0 if self.device == "cuda" and torch.cuda.is_available() else self.device
                    )
                    model = pipeline(
                        task="automatic-speech-recognition",
                        model=self.model_id,
                        device=pipeline_device,
                    )
                    _SHARED_WHISPER_MODELS[cache_key] = model
                    self._pipeline = model

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        return_timestamps: bool = True,
        speaker_id: str = "SPEAKER_00",
        speaker_name: str = "Nguoi noi",
    ) -> Tuple[str, List[TranscriptSegment]]:
        """
        Boc bang am thanh tieng Viet bang faster-whisper (int8 + Silero VAD)
        hoac callable pipeline (tuong thich 100% voi MockPipeline trong unit tests).
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

        elif isinstance(audio_input, np.ndarray):
            audio_data = audio_input
            if len(audio_data) == 0:
                raise ValueError("Mang du lieu am thanh audio_input rong (0 samples).")

            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            total_duration = len(audio_data) / float(sample_rate)
        else:
            raise ValueError(
                f"Dinh dang audio_input khong hop le: {type(audio_input)}. Chi chap nhan str, Path hoac np.ndarray."
            )

        start_infer = time.time()

        # Nhánh 1: faster-whisper WhisperModel (co phuong thuc .transcribe)
        if hasattr(self._pipeline, "transcribe"):
            audio_16k = _resample_to_16khz(audio_data, sample_rate)
            # Bat Silero VAD (vad_filter=True) khi boc bang file audio (return_timestamps=True);
            # voi turn ngan da cat san tu diarization (return_timestamps=False) thi khong cat VAD lan 2
            use_vad = bool(return_timestamps)

            with self._lock:
                try:
                    seg_iter, _info = self._pipeline.transcribe(
                        audio_16k,
                        language="vi",
                        beam_size=self.beam_size,
                        vad_filter=use_vad,
                        vad_parameters={"min_silence_duration_ms": self.min_silence_duration_ms}
                        if use_vad
                        else None,
                        without_timestamps=not return_timestamps,
                        condition_on_previous_text=False,
                    )
                    raw_fw_segments = list(seg_iter)

                    # Neu VAD loc mat turn ngan (< 10s), thu lai khong VAD de khong bo sot cau thoai
                    if not raw_fw_segments and use_vad and (total_duration or 0.0) <= 12.0:
                        seg_iter_retry, _ = self._pipeline.transcribe(
                            audio_16k,
                            language="vi",
                            beam_size=self.beam_size,
                            vad_filter=False,
                            without_timestamps=not return_timestamps,
                            condition_on_previous_text=False,
                        )
                        raw_fw_segments = list(seg_iter_retry)
                finally:
                    if self.device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            segments: List[TranscriptSegment] = []
            text_parts: List[str] = []

            for fw_seg in raw_fw_segments:
                seg_text = str(getattr(fw_seg, "text", "") or "").strip()
                if not seg_text:
                    continue
                text_parts.append(seg_text)

                st_val = getattr(fw_seg, "start", None)
                et_val = getattr(fw_seg, "end", None)
                start_sec = round(float(st_val), 2) if st_val is not None else 0.0
                if et_val is not None:
                    raw_end = float(et_val)
                    end_sec: Optional[float] = (
                        round(min(total_duration, raw_end), 2)
                        if total_duration is not None
                        else round(raw_end, 2)
                    )
                else:
                    end_sec = round(total_duration, 2) if total_duration is not None else None

                segments.append(
                    TranscriptSegment(
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        start_time=start_sec,
                        end_time=end_sec,
                        text=seg_text,
                    )
                )

            full_text = " ".join(text_parts).strip()
            logger.info(
                "faster-whisper (%s/%s) boc bang hoan tat %.2fs audio trong %.2fs (%d segments).",
                self.device,
                self.compute_type,
                total_duration or 0.0,
                time.time() - start_infer,
                len(segments),
            )
            return full_text, segments

        # Nhánh 2: Callable pipeline (MockPipeline trong unit tests hoac transformers fallback)
        inputs = {"raw": audio_data, "sampling_rate": sample_rate}
        with self._lock:
            try:
                if return_timestamps:
                    result = self._pipeline(inputs, return_timestamps=True)
                else:
                    result = self._pipeline(inputs)
            finally:
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        full_text = (
            result.get("text", "").strip()
            if isinstance(result, dict)
            else str(result or "").strip()
        )
        segments = []
        raw_chunks = result.get("chunks", []) if isinstance(result, dict) else []

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
