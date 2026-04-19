from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.model_config import ModelConfig
from models.rag_result import RagResult
from rag.answer_merger import AnswerMerger
from rag.citation_formatter import CitationFormatter
from rag.context_budget import ContextBudgetManager
from rag.model_client import ModelClient
from rag.prompt_builder import PromptBuilder
from rag.retriever import Retriever


@dataclass
class RagAgentInput:
    """
    Input payload for a RAG query.
    """

    question: str
    repo_name: str | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    top_k: int | None = None


class RagAgent:
    """
    Coordinates retrieval and model reasoning for repository questions.
    """

    def __init__(
        self,
        retriever: Retriever,
        model_client: ModelClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        context_budget_manager: ContextBudgetManager | None = None,
        citation_formatter: CitationFormatter | None = None,
        answer_merger: AnswerMerger | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or ModelConfig.from_env()
        self.model_client = model_client or ModelClient(self.config)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.context_budget_manager = context_budget_manager or ContextBudgetManager(
            max_context_chars=self.config.max_context_chars
        )
        self.citation_formatter = citation_formatter or CitationFormatter()
        self.answer_merger = answer_merger or AnswerMerger()

    def run(self, payload: RagAgentInput) -> RagResult:
        top_k = payload.top_k or self.config.top_k
        retrieval = self.retriever.retrieve(payload.question, top_k=top_k)

        budget_result = self.context_budget_manager.apply(retrieval.chunks)
        selected_chunks = budget_result.selected_chunks

        prompt = self.prompt_builder.build(
            question=payload.question,
            chunks=selected_chunks,
            repo_name=payload.repo_name,
            findings=payload.findings,
        )

        model_response = self.model_client.complete(prompt)
        parsed = self._parse_response_text(model_response.text)

        rag_result = RagResult(
            answer=parsed["answer"],
            suggestions=parsed["suggestions"],
            citations=[self.citation_formatter.format_chunk(chunk).__dict__ for chunk in selected_chunks],
            follow_up_questions=parsed["follow_up_questions"],
            notes=parsed["notes"],
            raw_response=model_response.raw_response,
        )

        return rag_result

    def _parse_response_text(self, text: str) -> dict[str, Any]:
        """
        Parse model output into a structured result.

        Expected format:
        - answer: ...
        - suggestions:
          - ...
        - citations:
          - ...
        - follow_up_questions:
          - ...
        - notes:
          - ...
        """
        sections: dict[str, list[str]] = {
            "answer": [],
            "suggestions": [],
            "citations": [],
            "follow_up_questions": [],
            "notes": [],
        }

        current_section = "answer"

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            lowered = line.lower().rstrip(":")
            if lowered in sections:
                current_section = lowered
                continue

            if line.startswith("- "):
                sections[current_section].append(line[2:].strip())
            else:
                sections[current_section].append(line)

        return {
            "answer": "\n".join(sections["answer"]).strip(),
            "suggestions": sections["suggestions"],
            "citations": sections["citations"],
            "follow_up_questions": sections["follow_up_questions"],
            "notes": sections["notes"],
        }