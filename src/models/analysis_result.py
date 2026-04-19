from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    """
    Final structured result returned by the orchestrator.
    """

    repo_name: str
    summary: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    rag_answer: str | None = None
    rag_suggestions: list[str] = field(default_factory=list)
    rag_citations: list[dict[str, Any]] = field(default_factory=list)
    rag_follow_up_questions: list[str] = field(default_factory=list)
    rag_notes: list[str] = field(default_factory=list)