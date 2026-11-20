"""
Module: pyannote_diarizer.py
Duong dan: src/core/ai/diarization/pyannote_diarizer.py
Muc dich: Tu dong phan chia luot nguoi noi (Speaker Diarization) bang mo hinh
'pyannote/speaker-diarization-3.1' voi co che Lazy Loading, Singleton Caching
va tu dong tang toc tren GPU NVIDIA (CUDA) neu kha dung.
"""

import functools
import inspect
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import soundfile as sf

# Dam bao PyTorch 2.6+ nap duoc checkpoint pyannote.audio ma khong bi chan boi weights_only=True
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch

logger = logging.getLogger(__name__)


def _wrap_use_auth_token_compat(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Boc ham huggingface_hub de tu chuyen use_auth_token -> token neu khong con tham so cu."""
    if getattr(orig_fn, "_patched_use_auth_token", False):
        return orig_fn

    try:
        sig = inspect.signature(orig_fn)
    except (TypeError, ValueError):
        return orig_fn

    if "use_auth_token" in sig.parameters:
        return orig_fn

    @functools.wraps(orig_fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if "use_auth_token" in kwargs:
            auth_val = kwargs.pop("use_auth_token")
            if kwargs.get("token") is None:
                kwargs["token"] = auth_val
        return orig_fn(*args, **kwargs)

    setattr(_wrapped, "_patched_use_auth_token", True)
    return _wrapped


def _patch_hf_hub_compat() -> None:
    """
    Tuong thich nguoc cho pyannote.audio==3.3.2 khi chay voi huggingface_hub ban moi
    da loai bo tham so 'use_auth_token' tren hf_hub_download / snapshot_download.
    """
    try:
        import huggingface_hub
    except ImportError:
        return

    patched_map: Dict[str, Callable[..., Any]] = {}

    # 1. Patch huggingface_hub.hf_hub_download & huggingface_hub.file_download.hf_hub_download
    hf_dl = getattr(huggingface_hub, "hf_hub_download", None)
    if callable(hf_dl):
        wrapped_dl = _wrap_use_auth_token_compat(hf_dl)
        if wrapped_dl is not hf_dl or getattr(wrapped_dl, "_patched_use_auth_token", False):
            huggingface_hub.hf_hub_download = wrapped_dl
            patched_map["hf_hub_download"] = wrapped_dl
            try:
                import huggingface_hub.file_download as hf_file_download

                hf_file_download.hf_hub_download = wrapped_dl
            except Exception:
                pass

    # 2. Patch huggingface_hub.snapshot_download & huggingface_hub._snapshot_download.snapshot_download
    snap_dl = getattr(huggingface_hub, "snapshot_download", None)
    if callable(snap_dl):
        wrapped_snap = _wrap_use_auth_token_compat(snap_dl)
        if wrapped_snap is not snap_dl or getattr(wrapped_snap, "_patched_use_auth_token", False):
            huggingface_hub.snapshot_download = wrapped_snap
            patched_map["snapshot_download"] = wrapped_snap
            try:
                import huggingface_hub._snapshot_download as hf_snap_mod

                hf_snap_mod.snapshot_download = wrapped_snap
            except Exception:
                pass

    if not patched_map:
        return

    # 3. Quet sys.modules de cap nhat cho moi module pyannote.* da lo import san truoc do
    for mod_name, mod_obj in list(sys.modules.items()):
        if mod_obj is None:
            continue
        if mod_name == "pyannote" or mod_name.startswith("pyannote."):
            for attr_name, wrapped_fn in patched_map.items():
                if hasattr(mod_obj, attr_name):
                    try:
                        setattr(mod_obj, attr_name, wrapped_fn)
                    except Exception:
                        pass


class PyannoteSpeakerDiarizer:
    """
    Bo phan chia nguoi noi su dung Pipeline 'pyannote/speaker-diarization-3.1'.
    Ap dung Lazy Loading + Singleton Caching va tu dong chuyen sang torch.device('cuda') neu co GPU.
    """

    MODEL_ID = "pyannote/speaker-diarization-3.1"

    def __init__(self, hf_token: Optional[str] = None) -> None:
        self._hf_token = (
            hf_token
            if hf_token is not None
            else os.getenv("HF_TOKEN", "")
        ).strip().strip("\"'").strip()
        self._pipeline: Any = None
        self._lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _resolve_token(self, hf_token: Optional[str] = None) -> str:
        token = (
            hf_token
            if hf_token is not None
            else (self._hf_token or os.getenv("HF_TOKEN", ""))
        )
        clean_token = str(token or "").strip().strip("\"'").strip()
        if clean_token in ("", "your_huggingface_token_here"):
            return ""
        return clean_token

    def load_model(self, hf_token: Optional[str] = None) -> None:
        """Nap Pipeline pyannote/speaker-diarization-3.1 (thread-safe double-checked locking)."""
        if self._pipeline is not None:
            return

        token = self._resolve_token(hf_token)
        if not token:
            raise ValueError(
                "Chua cau hinh HF_TOKEN hop le de tai mo hinh pyannote/speaker-diarization-3.1."
            )

        with self._lock:
            if self._pipeline is not None:
                return

            _patch_hf_hub_compat()
            from pyannote.audio import Pipeline

            _patch_hf_hub_compat()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(
                "Dang nap mo hinh '%s' len thiet bi '%s'...",
                self.MODEL_ID,
                self.device,
            )
            start_t = time.time()

            pipeline = Pipeline.from_pretrained(
                self.MODEL_ID,
                use_auth_token=token,
            )
            if pipeline is None:
                raise RuntimeError(
                    f"Khong the khoi tao Pipeline '{self.MODEL_ID}'. Kiem tra quyen truy cap HF_TOKEN."
                )

            pipeline.to(self.device)
            self._pipeline = pipeline

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            logger.info(
                "Nap mo hinh '%s' thanh cong tren '%s' trong %.2f giay.",
                self.MODEL_ID,
                self.device,
                time.time() - start_t,
            )

    def diarize(
        self,
        audio_path: Union[str, Path],
        hf_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chay Speaker Diarization tren file am thanh va tra ve danh sach luot noi:
        [{"speaker": "SPEAKER_00", "speaker_id": "SPEAKER_00", "start": 0.45, "end": 5.12}, ...]
        """
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Khong tim thay file am thanh de phan vai: {path_obj.resolve()}"
            )

        self.load_model(hf_token=hf_token)

        # Doc waveform bang soundfile de tranh loi backend torchaudio tren cac he dieu hanh khac nhau
        audio_data, sample_rate = sf.read(str(path_obj), dtype="float32", always_2d=True)
        if len(audio_data) == 0:
            raise ValueError(f"File am thanh '{path_obj}' rong (0 samples).")

        # Chuyen ve (channels, samples) theo chuan pyannote.audio in-memory input
        mono = np.mean(audio_data, axis=1, keepdims=True).T
        waveform_tensor = torch.from_numpy(mono.astype(np.float32))
        audio_in_memory = {
            "waveform": waveform_tensor,
            "sample_rate": int(sample_rate),
        }

        logger.info(
            "Bat dau phan chia nguoi noi bang '%s' tren '%s' cho file '%s'...",
            self.MODEL_ID,
            self.device,
            path_obj.name,
        )
        start_t = time.time()

        with self._lock:
            try:
                diarization_result = self._pipeline(audio_in_memory)
            finally:
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        turns: List[Dict[str, Any]] = []
        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
            spk = str(speaker_label or "SPEAKER_00").strip()
            st = round(float(turn.start), 3)
            et = round(float(turn.end), 3)
            if et <= st:
                continue
            turns.append(
                {
                    "speaker": spk,
                    "speaker_id": spk,
                    "start": st,
                    "end": et,
                    "duration": round(et - st, 3),
                }
            )

        logger.info(
            "Hoan tat pyannote diarization trong %.2f giay: thu duoc %d luot noi.",
            time.time() - start_t,
            len(turns),
        )
        return turns


# Singleton instance
_diarizer_instance: Optional[PyannoteSpeakerDiarizer] = None
_diarizer_singleton_lock = threading.Lock()


def get_pyannote_diarizer(hf_token: Optional[str] = None) -> PyannoteSpeakerDiarizer:
    """Lay Singleton PyannoteSpeakerDiarizer."""
    global _diarizer_instance
    if _diarizer_instance is None:
        with _diarizer_singleton_lock:
            if _diarizer_instance is None:
                _diarizer_instance = PyannoteSpeakerDiarizer(hf_token=hf_token)
    return _diarizer_instance


def diarize_audio_with_pyannote(
    audio_path: Union[str, Path],
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ham tien ich goi singleton PyannoteSpeakerDiarizer.diarize()."""
    diarizer = get_pyannote_diarizer(hf_token=hf_token)
    return diarizer.diarize(audio_path=audio_path, hf_token=hf_token)
