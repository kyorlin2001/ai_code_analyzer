from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models.retrieval_chunk import RetrievalChunk


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Configuration for line-based chunking.
    """

    chunk_size_lines: int = 80
    overlap_lines: int = 15
    max_file_lines: int = 2_000


class Chunker:
    """
    Splits raw file content into retrieval chunks.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_text(
        self,
        text: str,
        file_path: str,
        language: str | None = None,
    ) -> list[RetrievalChunk]:
        lines = text.splitlines()
        if not lines:
            return []

        if len(lines) > self.config.max_file_lines:
            lines = lines[: self.config.max_file_lines]

        chunks: list[RetrievalChunk] = []
        step = max(1, self.config.chunk_size_lines - self.config.overlap_lines)

        start = 0
        chunk_index = 0

        while start < len(lines):
            end = min(len(lines), start + self.config.chunk_size_lines)
            chunk_lines = lines[start:end]

            if not chunk_lines:
                break

            chunk_text = "\n".join(chunk_lines).strip()
            if chunk_text:
                chunk_id = f"{file_path}::chunk-{chunk_index}"
                chunks.append(
                    RetrievalChunk(
                        text=chunk_text,
                        file_path=file_path,
                        chunk_id=chunk_id,
                        language=language,
                        start_line=start + 1,
                        end_line=end,
                    )
                )

            chunk_index += 1
            start += step

        return chunks

    def chunk_many(
        self,
        items: Iterable[tuple[str, str, str | None]],
    ) -> list[RetrievalChunk]:
        """
        Chunk many files at once.

        Each item is expected to be:
        (text, file_path, language)
        """
        chunks: list[RetrievalChunk] = []
        for text, file_path, language in items:
            chunks.extend(self.chunk_text(text=text, file_path=file_path, language=language))
        return chunks