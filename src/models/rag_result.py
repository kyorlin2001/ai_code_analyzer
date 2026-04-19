from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RagResult:
    """
    Structured output from a RAG-based model response.
    """

    answer: str
    suggestions: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None