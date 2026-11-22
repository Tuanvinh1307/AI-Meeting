"""
Module: state.py
Duong dan: src/core/ai/graph/state.py
Muc dich: Dinh nghia MeetingGraphState (TypedDict) cho LangGraph Pipeline.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from src.core.ai.asr.phowhisper import PhoWhisperASR
from src.core.ai.diarization.alignment import DiarizationInputType
from src.core.ai.llm.gemini_client import MeetingLLMAnalyzer
from src.core.ai.llm.grounding import LLMExtractionOutput
from src.schemas.meeting import TranscriptSegment


class MeetingGraphState(TypedDict, total=False):
    """Trang thai trung tam cua LangGraph Pipeline xu ly cuoc hop End-to-End."""

    audio_path: Optional[Union[str, Path]]
    valid_speakers: List[Any]
    raw_segments: List[TranscriptSegment]
    aligned_segments: List[TranscriptSegment]
    transcript_text: str
    analysis_result: Optional[LLMExtractionOutput]
    validation_errors: List[str]
    retry_count: int
    # Cac tham so dieu phoi runtime tuy chon
    meeting_id: Optional[int]
    progress_callback: Optional[Callable[[str, int], None]]
    asr: Optional[PhoWhisperASR]
    analyzer: Optional[MeetingLLMAnalyzer]
    diarization_input: Optional[DiarizationInputType]
    diarization_segments: Optional[DiarizationInputType]
    diarization_turns: Optional[List[Dict[str, Any]]]
    speaker_mapping: Optional[Dict[str, str]]
    run_llm_analysis: bool
