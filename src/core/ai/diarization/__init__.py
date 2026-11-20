"""
Diarization & Voiceprint Identification Package.
"""

from src.core.ai.diarization.alignment import (
    DiarizationInputType,
    align_segments_by_overlap,
    clean_and_merge_diarization_turns,
    load_diarization_segments,
    transcribe_with_diarization,
)
from src.core.ai.diarization.pyannote_diarizer import (
    PyannoteSpeakerDiarizer,
    diarize_audio_with_pyannote,
    get_pyannote_diarizer,
)
from src.core.ai.diarization.voiceprint import (
    auto_diarize_and_identify_segments,
    cosine_similarity,
    extract_voiceprint_embedding,
    match_speakers_to_profiles,
    save_audio_clip,
)

__all__ = [
    "DiarizationInputType",
    "PyannoteSpeakerDiarizer",
    "align_segments_by_overlap",
    "auto_diarize_and_identify_segments",
    "clean_and_merge_diarization_turns",
    "cosine_similarity",
    "diarize_audio_with_pyannote",
    "extract_voiceprint_embedding",
    "get_pyannote_diarizer",
    "load_diarization_segments",
    "match_speakers_to_profiles",
    "save_audio_clip",
    "transcribe_with_diarization",
]
