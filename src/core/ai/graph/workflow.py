"""
Module: workflow.py
Duong dan: src/core/ai/graph/workflow.py
Muc dich: Lap rap LangGraph StateGraph (transcribe -> speaker_align -> analyze -> validate)
va export meeting_graph.
"""

import logging

from langgraph.graph import END, START, StateGraph

from src.core.ai.graph.nodes import (
    analyze_node,
    speaker_align_node,
    transcribe_node,
    validate_node,
)
from src.core.ai.graph.state import MeetingGraphState

logger = logging.getLogger(__name__)


def _route_after_validate(state: MeetingGraphState) -> str:
    """
    Conditional Edge sau validate_node:
    - Neu ket qua den tu phuong an du phong offline (used_fallback=True), bo qua retry va di toi END.
    - Neu co loi (validation_errors khac rong) va retry_count < 2 thi vong lai analyze_node;
      nguoc lai di toi END.
    """
    if state.get("run_llm_analysis") is False:
        return END

    result = state.get("analysis_result")
    if result is not None and getattr(result, "used_fallback", False):
        return END

    errors = state.get("validation_errors") or []
    retry_count = int(state.get("retry_count") or 0)
    if errors and retry_count < 2:
        logger.warning(
            "Phat hien %d loi validation (retry_count=%d < 2). Vong lai analyze_node...",
            len(errors),
            retry_count,
        )
        return "analyze_node"
    return END


def build_meeting_graph():
    """Xay dung va bien dich LangGraph StateGraph hoan chinh cho quy trinh xu ly cuoc hop."""
    workflow = StateGraph(MeetingGraphState)

    workflow.add_node("transcribe_node", transcribe_node)
    workflow.add_node("speaker_align_node", speaker_align_node)
    workflow.add_node("analyze_node", analyze_node)
    workflow.add_node("validate_node", validate_node)

    workflow.add_edge(START, "transcribe_node")
    workflow.add_edge("transcribe_node", "speaker_align_node")
    workflow.add_edge("speaker_align_node", "analyze_node")
    workflow.add_edge("analyze_node", "validate_node")
    workflow.add_conditional_edges(
        "validate_node",
        _route_after_validate,
        {
            "analyze_node": "analyze_node",
            END: END,
        },
    )

    return workflow.compile()


meeting_graph = build_meeting_graph()
