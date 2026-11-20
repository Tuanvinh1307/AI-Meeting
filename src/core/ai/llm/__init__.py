"""
LLM Analysis Package (Prompts, Grounding, Fallback, Gemini Client).
"""

from src.core.ai.llm.fallback import rule_based_analyze
from src.core.ai.llm.gemini_client import MeetingLLMAnalyzer
from src.core.ai.llm.grounding import (
    LLMExtractionOutput,
    _clean_asr_vietnamese_text,
    _find_best_matching_segment,
    build_cleaned_segments,
    clean_vietnamese_asr_text,
    extract_valid_speaker_names,
    is_assignee_valid,
    validate_and_ground_tasks,
)
from src.core.ai.llm.prompts import (
    build_formatted_turns,
    build_system_prompt,
    build_user_prompt_with_feedback,
)

__all__ = [
    "LLMExtractionOutput",
    "MeetingLLMAnalyzer",
    "_clean_asr_vietnamese_text",
    "_find_best_matching_segment",
    "build_cleaned_segments",
    "build_formatted_turns",
    "build_system_prompt",
    "build_user_prompt_with_feedback",
    "clean_vietnamese_asr_text",
    "extract_valid_speaker_names",
    "is_assignee_valid",
    "rule_based_analyze",
    "validate_and_ground_tasks",
]
