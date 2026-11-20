"""
ASR Package (PhoWhisper Speech-to-Text).
"""

from src.core.ai.asr.phowhisper import PhoWhisperASR, split_audio_by_silence

__all__ = ["PhoWhisperASR", "split_audio_by_silence"]
