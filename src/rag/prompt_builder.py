from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.retrieval_chunk import RetrievalChunk


@dataclass(frozen=True)
class PromptBundle:
    """
    Structured prompt payload for the hosted model.
    """

    system_prompt: str
    user_prompt: str


class PromptBuilder:
    """
    Builds prompts for RAG-based repository analysis.
    """

    def build(
        self,
        question: str,
        chunks: list[RetrievalChunk],
        repo_name: str | None = None,
        findings: list[dict[str, Any]] | None = None,
    ) -> PromptBundle:
        system_prompt = (
            "You are an expert software analysis assistant.\n"
            "Answer using the repository context provided below.\n"
            "If the context is insufficient, say what is missing.\n"
            "Be specific and practical.\n"
            "Reference files or chunks when helpful."
        )

        user_prompt = self._build_user_prompt(
            question=question,
            chunks=chunks,
            repo_name=repo_name,
            findings=findings or [],
        )

        return PromptBundle(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _build_user_prompt(
        self,
        question: str,
        chunks: list[RetrievalChunk],
        repo_name: str | None,
        findings: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []

        lines.append(f"Question: {question}")

        if repo_name:
            lines.append(f"Repository: {repo_name}")

        if findings:
            lines.append("")
            lines.append("Existing analysis findings:")
            for finding in findings:
                severity = finding.get("severity", "info")
                message = finding.get("message", "")
                lines