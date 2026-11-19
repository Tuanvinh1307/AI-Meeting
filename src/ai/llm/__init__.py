"""
Module: src.ai.llm
Cung cap cac thanh phan phan tich cuoc hop bang LLM (Large Language Model)
kem co che hau kiem bang chung (evidence grounding) va fallback theo quy tac.
"""

from src.ai.llm.analyzer import (
    LLMExtractionOutput,
    MeetingLLMAnalyzer,
    clean_vietnamese_asr_text,
    validate_and_ground_tasks,
)

__all__ = [
    "LLMExtractionOutput",
    "MeetingLLMAnalyzer",
    "clean_vietnamese_asr_text",
    "validate_and_ground_tasks",
]
