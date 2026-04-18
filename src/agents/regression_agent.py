from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.analysis_state import AnalysisState


@dataclass
class RegressionOutput:
    """
    Result of comparing the current analysis to a baseline.
    """

    new_findings: list[dict[str, Any]]
    resolved_findings: list[dict[str, Any]]
    notes: list[str]


class RegressionAgent:
    """
    Compares current findings against a previous baseline.
    """

    def run(self, state: AnalysisState) -> RegressionOutput:
        baseline = state.baseline_findings or []
        current = state.findings

        baseline_messages = {f.get("message") for f in baseline if isinstance(f, dict)}
        current_messages = {f.get("message") for f in current if isinstance(f, dict)}

        new_findings = [f for f in current if f.get("message") not in baseline_messages]
        resolved_findings = [f for f in baseline if f.get("message") not in current_messages]

        notes: list[str] = []
        if new_findings:
            notes.append(f"Detected {len(new_findings)} new finding(s).")
        if resolved_findings:
            notes.append(f"Detected {len(resolved_findings)} resolved finding(s).")
        if not notes:
            notes.append("No meaningful regression changes detected.")

        output = RegressionOutput(
            new_findings=new_findings,
            resolved_findings=resolved_findings,
            notes=notes,
        )

        state.agent_outputs["regression"] = output
        state.summaries["regression"] = self._format_summary(output)
        return output

    def _format_summary(self, output: RegressionOutput) -> str:
        lines = ["Regression analysis:"]
        for note in output.notes:
            lines.append(f"- {note}")

        if output.new_findings:
            lines.append("New findings:")
            for finding in output.new_findings:
                lines.append(f"- {finding.get('message', '')}")

        if output.resolved_findings:
            lines.append("Resolved findings:")
            for finding in output.resolved_findings:
                lines.append(f"- {finding.get('message', '')}")

        return "\n".join(lines)