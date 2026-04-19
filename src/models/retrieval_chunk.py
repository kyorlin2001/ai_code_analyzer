from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalChunk:
    """
    A small retrievable unit of repository content.
    """

    text: str
    file_path: str
    chunk_id: str
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    score: float = field(default=0.0, compare=False)