from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.analysis_state import AnalysisState


@dataclass
class DependencyOutput:
    """
    Result of dependency analysis.
    """

    imports_by_language: dict[str, int]
    package_manifests: list[str]
    notes: list[str]
    findings: list[dict[str, Any]]


class DependencyAgent:
    """
    Performs lightweight dependency analysis.

    Responsibilities:
    - detect dependency manifest files
    - estimate import/package usage
    - produce early findings for downstream reporting
    """

    MANIFEST_FILES = {
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "Pipfile.lock",
        "setup.py",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "go.mod",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "composer.json",
    }

    def run(self, state: AnalysisState) -> DependencyOutput:
        package_manifests = self._find_manifests(state.files)
        imports_by_language = self._estimate_imports(state.files)

        findings: list[dict[str, Any]] = []
        notes: list[str] = []

        if package_manifests:
            notes.append(f"Found {len(package_manifests)} dependency manifest file(s).")
        else:
            notes.append("No common dependency manifest files were detected.")

        if "Python" in imports_by_language:
            findings.append(
                {
                    "severity": "info",
                    "message": "Python source files detected; check for unused or stale dependencies.",
                }
            )

        output = DependencyOutput(
            imports_by_language=imports_by_language,
            package_manifests=package_manifests,
            notes=notes,
            findings=findings,
        )

        state.summaries["dependencies"] = self._format_summary(output)
        state.agent_outputs["dependencies"] = output
        state.findings.extend(findings)
        return output

    def _find_manifests(self, files: list[str]) -> list[str]:
        return [f for f in files if Path(f).name in self.MANIFEST_FILES]

    def _estimate_imports(self, files: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}

        for file_path in files:
            suffix = Path(file_path).suffix.lower()
            if suffix == ".py":
                counts["Python"] = counts.get("Python", 0) + 1
            elif suffix in {".js", ".jsx"}:
                counts["JavaScript"] = counts.get("JavaScript", 0) + 1
            elif suffix in {".ts", ".tsx"}:
                counts["TypeScript"] = counts.get("TypeScript", 0) + 1
            elif suffix in {".java", ".kt"}:
                counts["JVM"] = counts.get("JVM", 0) + 1
            elif suffix == ".go":
                counts["Go"] = counts.get("Go", 0) + 1
            elif suffix == ".rs":
                counts["Rust"] = counts.get("Rust", 0) + 1

        return counts

    def _format_summary(self, output: DependencyOutput) -> str:
        lines = ["Dependency analysis:"]
        if output.package_manifests:
            lines.append("Manifests:")
            for manifest in output.package_manifests:
                lines.append(f"- {manifest}")
        else:
            lines.append("No dependency manifests found.")

        if output.imports_by_language:
            lines.append("Source file counts by language:")
            for language, count in sorted(output.imports_by_language.items()):
                lines.append(f"- {language}: {count}")

        lines.extend(output.notes)
        return "\n".join(lines)