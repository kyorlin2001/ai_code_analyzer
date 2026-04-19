from __future__ import annotations

from dataclasses import dataclass, field

from models.retrieval_chunk import RetrievalChunk


@dataclass(frozen=True)
class ContextBudgetResult:
    """
    Result of selecting chunks within a context budget.
    """

    selected_chunks: list[RetrievalChunk] = field(default_factory=list)
    total_characters: int = 0
    truncated: bool = False


class ContextBudgetManager:
    """
    Keeps retrieved context within a maximum character budget.
    """

    def __init__(self, max_context_chars: int = 24_000) -> None:
        self.max_context_chars = max_context_chars

    def apply(self, chunks: list[RetrievalChunk]) -> ContextBudgetResult:
        selected: list[RetrievalChunk] = []
        total_chars = 0
        truncated = False

        for chunk in chunks:
            chunk_length = len(chunk.text)

            if selected and total_chars + chunk_length > self.max_context_chars:
                truncated = True
                break

            if not selected and chunk_length > self.max_context_chars:
                # Always keep at least one chunk, even if it exceeds the budget.
                selected.append(chunk)
                total_chars += chunk_length
                truncated = True
                break

            selected.append(chunk)
            total_chars += chunk_length

        if len(selected) < len(chunks):
            truncated = True

        return ContextBudgetResult(
            selected_chunks=selected,
            total_characters=total_chars,
            truncated=truncated,
        )