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
            "Use the repository context provided in the user prompt.\n"
            "If the context is insufficient, say exactly what is missing.\n"
            "Be specific, practical, and grounded in the evidence.\n"
            "Prefer file names, code snippets, and concrete suggestions."
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
        #just a test remove later.
        question = "Find potential issuues in the codebase"

        lines.append(f"Question: {question}")

        if repo_name:
            lines.append(f"Repository: {repo_name}")

        if findings:
            lines.append("")
            lines.append("Existing analysis findings:")
            for finding in findings:
                severity = finding.get("severity", "info")
                message = finding.get("message", "")
                lines.append(f"- [{severity}] {message}")

        lines.append("")
        lines.append("Repository context:")

        if not chunks:
            lines.append("- No relevant chunks were retrieved.")
        else:
            for index, chunk in enumerate(chunks, start=1):
                lines.append("")
                lines.append(f"Context block {index}:")
                lines.append(f"File: {chunk.file_path}")
                if chunk.start_line is not None and chunk.end_line is not None:
                    lines.append(f"Lines: {chunk.start_line}-{chunk.end_line}")
                if chunk.language:
                    lines.append(f"Language: {chunk.language}")
                lines.append("Content:")
                lines.append(chunk.text)

        lines.append("")
        lines.append("Instructions:")
        lines.append("- Answer using the repository context above.")
        lines.append("- If you suggest improvements, make them specific and actionable.")
        lines.append("- If the context is insufficient, explain what is missing.")
        lines.append("- Do not ask for repository files again unless no context was retrieved.")
        return "\n".join(lines)