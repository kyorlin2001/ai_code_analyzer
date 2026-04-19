from __future__ import annotations

from dataclasses import dataclass, field

from models.rag_result import RagResult


@dataclass
class MergedAnswer:
    """
    Combined output from the deterministic analyzer and the RAG layer.
    """

    summary: str
    findings: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    rag_answer: str | None = None
    rag_suggestions: list[str] = field(default_factory=list)
    rag_citations: list[dict] = field(default_factory=list)
    rag_follow_up_questions: list[str] = field(default_factory=list)
    rag_notes: list[str] = field(default_factory=list)


class AnswerMerger:
    """
    Merges existing analysis output with RAG-generated insights.
    """

    def merge(
        self,
        summary: str,
        findings: list[dict],
        recommendations: list[str],
        rag_result: RagResult | None = None,
    ) -> MergedAnswer:
        if rag_result is None:
            return MergedAnswer(
                summary=summary,
                findings=list(findings),
                recommendations=list(recommendations),
            )

        merged_summary = self._append_rag_summary(summary, rag_result)

        merged_recommendations = list(recommendations)
        for suggestion in rag_result.suggestions:
            if suggestion not in merged_recommendations:
                merged_recommendations.append(suggestion)

        return MergedAnswer(
            summary=merged_summary,
            findings=list(findings),
            recommendations=merged_recommendations,
            rag_answer=rag_result.answer,
            rag_suggestions=list(rag_result.suggestions),
            rag_citations=list(rag_result.citations),
            rag_follow_up_questions=list(rag_result.follow_up_questions),
            rag_notes=list(rag_result.notes),
        )

    def _append_rag_summary(self, summary: str, rag_result: RagResult) -> str:
        parts = [summary]

        if rag_result.answer:
            parts.append("")
            parts.append("RAG Insight:")
            parts.append(rag_result.answer)

        if rag_result.notes:
            parts.append("")
            parts.append("RAG Notes:")
            for note in rag_result.notes:
                parts.append(f"- {note}")

        return "\n".join(parts)