from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.analysis_state import AnalysisState


@dataclass
class IntakeOutput:
    """
    Result of the intake agent.
    """

    repo_name: str
    file_count: int
    language_summary: dict[str, int]
    notes: list[str]


class IntakeAgent:
    """
    Performs the first-pass repository intake.

    Responsibilities:
    - inspect repository contents
    - estimate language distribution
    - capture basic notes for downstream agents
    """

    def run(self, state: AnalysisState) -> IntakeOutput:
        language_summary = self._summarize_languages(state.files)

        notes: list[str] = []
        if not state.files:
            notes.append("No supported text files were found in the repository.")
        else:
            notes.append(f"Found {len(state.files)} supported text files.")

        output = IntakeOutput(
            repo_name=state.repo_name,
            file_count=len(state.files),
            language_summary=language_summary,
            notes=notes,
        )

        state.summaries["intake"] = self._format_summary(output)
        state.agent_outputs["intake"] = output
        return output

    def _summarize_languages(self, files: list[str]) -> dict[str, int]:
        """
        Very lightweight extension-based language summary.
        """
        extension_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript React",
            ".jsx": "JavaScript React",
            ".java": "Java",
            ".kt": "Kotlin",
            ".go": "Go",
            ".rs": "Rust",
        }

        language_summary: dict[str, int] = {}
        for file in files:
            for extension, language in extension_map.items():
                if file.endswith(extension):
                    language_summary[language] = language_summary.get(language, 0) + 1
                    break

        return language_summary

    def _format_summary(self, output: IntakeOutput) -> str:
        summary = f"Intake Agent Summary:\n"
        summary += f"  Repo Name: {output.repo_name}\n"
        summary += f"  File Count: {output.file_count}\n"
        summary += "  Language Summary:\n"
        for language, count in output.language_summary.items():
            summary += f"    {language}: {count}\n"
        summary += "  Notes:\n"
        for note in output.notes:
            summary += f"    {note}\n"
        return summary