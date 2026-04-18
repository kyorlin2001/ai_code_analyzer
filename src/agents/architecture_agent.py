from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.analysis_state import AnalysisState


@dataclass
class ArchitectureOutput:
    """
    Result of architecture analysis.
    """

    entry_points: list[str]
    module_groups: dict[str, list[str]]
    notes: list[str]
    findings: list[dict[str, Any]]


class ArchitectureAgent:
    """
    Performs a lightweight architecture pass over the repository.

    Responsibilities:
    - identify likely entry points
    - group files into rough modules/layers
    - produce architecture observations
    """

    ENTRY_POINT_NAMES = {
        "app.py",
        "main.py",
        "__init__.py",
        "index.js",
        "index.ts",
        "server.py",
        "run.py",
    }

    def run(self, state: AnalysisState) -> ArchitectureOutput:
        entry_points = self._find_entry_points(state.files)
        module_groups = self._group_files(state.files)

        findings: list[dict[str, Any]] = []
        notes: list[str] = []

        if entry_points:
            notes.append(f"Found {len(entry_points)} likely entry point(s).")
            findings.append(
                {
                    "severity": "info",
                    "message": "Likely application entry points were detected.",
                    "details": entry_points,
                }
            )
        else:
            notes.append("No obvious entry points were detected.")

        output = ArchitectureOutput(
            entry_points=entry_points,
            module_groups=module_groups,
            notes=notes,
            findings=findings,
        )

        state.summaries["architecture"] = self._format_summary(output)
        state.agent_outputs["architecture"] = output
        state.findings.extend(findings)
        return output

    def _find_entry_points(self, files: list[str]) -> list[str]:
        matches: list[str] = []
        for file_path in files:
            if Path(file_path).name in self.ENTRY_POINT_NAMES:
                matches.append(file_path)
        return matches

    def _group_files(self, files: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "root": [],
            "src": [],
            "tests": [],
            "config": [],
            "other": [],
        }

        for file_path in files:
            parts = Path(file_path).parts

            if len(parts) == 1:
                groups["root"].append(file_path)
            elif "test" in file_path.lower() or "tests" in parts:
                groups["tests"].append(file_path)
            elif parts[0] in {"src", "app", "lib"}:
                groups["src"].append(file_path)
            elif Path(file_path).suffix.lower() in {".toml", ".yaml", ".yml", ".json", ".ini"}:
                groups["config"].append(file_path)
            else:
                groups["other"].append(file_path)

        return {group: items for group, items in groups.items() if items}

    def _format_summary(self, output: ArchitectureOutput) -> str:
        lines = ["Architecture analysis:"]
        if output.entry_points:
            lines.append("Entry points:")
            for entry in output.entry_points:
                lines.append(f"- {entry}")
        else:
            lines.append("No obvious entry points found.")

        if output.module_groups:
            lines.append("Module groups:")
            for group, items in sorted(output.module_groups.items()):
                lines.append(f"- {group}: {len(items)} file(s)")

        lines.extend(output.notes)
        return "\n".join(lines)