from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.analysis_state import AnalysisState


@dataclass
class IssueOutput:
    """
    Result of issue / risk analysis.
    """

    issues: list[dict[str, Any]]
    notes: list[str]


class IssueAgent:
    """
    Performs a lightweight issue/risk scan over the repository.

    Responsibilities:
    - flag obvious maintenance risks
    - collect simple heuristics-based issues
    - prepare findings for the final report
    """

    def run(self, state: AnalysisState) -> IssueOutput:
        issues: list[dict[str, Any]] = []
        notes: list[str] = []

        if not state.files:
            issues.append(
                {
                    "severity": "medium",
                    "message": "Repository contains no supported source files.",
                }
            )
            notes.append("No source files available for deeper analysis.")
        else:
            issues.extend(self._scan_repo_heuristics(state.files))

        output = IssueOutput(issues=issues, notes=notes)

        state.summaries["issues"] = self._format_summary(output)
        state.agent_outputs["issues"] = output
        state.findings.extend(issues)
        return output

    def _scan_repo_heuristics(self, files: list[str]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        has_tests = any("test" in f.lower() or "tests" in Path(f).parts for f in files)
        has_readme = any(Path(f).name.lower() == "readme.md" for f in files)
        has_gitignore = any(Path(f).name == ".gitignore" for f in files)

        if not has_tests:
            issues.append(
                {
                    "severity": "medium",
                    "message": "No tests directory or test files were detected.",
                }
            )

        if not has_readme:
            issues.append(
                {
                    "severity": "low",
                    "message": "No README.md file was detected.",
                }
            )

        if not has_gitignore:
            issues.append(
                {
                    "severity": "low",
                    "message": "No .gitignore file was detected.",
                }
            )

        return issues

    def _format_summary(self, output: IssueOutput) -> str:
        lines = ["Issue analysis:"]
        if output.issues:
            for issue in output.issues:
                lines.append(f"- [{issue.get('severity', 'info')}] {issue.get('message', '')}")
        else:
            lines.append("No obvious issues detected.")

        lines.extend(output.notes)
        return "\n".join(lines)