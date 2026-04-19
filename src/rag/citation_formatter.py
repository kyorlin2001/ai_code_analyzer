from __future__ import annotations

from dataclasses import dataclass

from models.retrieval_chunk import RetrievalChunk


@dataclass(frozen=True)
class Citation:
    """
    Formatted citation for a retrieved chunk.
    """

    file_path: str
    chunk_id: str
    label: str
    start_line: int | None = None
    end_line: int | None = None


class CitationFormatter:
    """
    Formats retrieval chunks into lightweight citations.
    """

    def format_chunk(self, chunk: RetrievalChunk) -> Citation:
        if chunk.start_line is not None and chunk.end_line is not None:
            label = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        else:
            label = chunk.file_path

        return Citation(
            file_path=chunk.file_path,
            chunk_id=chunk.chunk_id,
            label=label,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )

    def format_chunks(self, chunks: list[RetrievalChunk]) -> list[Citation]:
        return [self.format_chunk(chunk) for chunk in chunks]