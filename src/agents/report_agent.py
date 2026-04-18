from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.analysis_state import AnalysisState


@dataclass
class ReportOutput:
    """
    Final report payload created from accumulated agent results.
    """

    summary: str
    findings: list[dict[str, Any]]
    recommendations: list[str]


class ReportAgent:
    """
    Combines the outputs of all agents into a final report.
    """

    def run(self, state: AnalysisState) -> ReportOutput:
        summary = self._build_summary(state)
        findings = list(state.findings)
        recommendations = self._build_recommendations(state)

        return ReportOutput(
            summary=summary,
            findings=findings,
            recommendations=recommendations,
        )

    def _build_summary(self, state: AnalysisState) -> str:
        lines = [f"Repository: {state.repo_name}"]

        if state.summaries.get("intake"):
            lines.append("")
            lines.append(state.summaries["intake"])

        if state.summaries.get("dependencies"):
            lines.append("")
            lines.append(state.summaries["dependencies"])

        if state.summaries.get("architecture"):
            lines.append("")
            lines.append(state.summaries["architecture"])

        if state.summaries.get("issues"):
            lines.append("")
            lines.append(state.summaries["issues"])

        return "\n".join(lines)

    def _build_recommendations(self, state: AnalysisState) -> list[str]:
        recommendations: list[str] = []

        issues = state.agent_outputs.get("issues")
        if issues and getattr(issues, "issues", None):
            recommendations.append("Address the identified repository risks first.")

        if not state.files:
            recommendations.append("Add supported source files or verify the repository path.")

        if not state.findings:
            recommendations.append("Run deeper code-level analysis for more specific findings.")

        if not recommendations:
            recommendations.append("Repository looks structurally healthy at a high level.")

        return recommendations