from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from models.retrieval_chunk import RetrievalChunk


@dataclass
class ChunkIndex:
    """
    In-memory index for repository retrieval chunks.
    """

    chunks: list[RetrievalChunk] = field(default_factory=list)

    def add_chunks(self, chunks: Iterable[RetrievalChunk]) -> None:
        self.chunks.extend(chunks)

    def clear(self) -> None:
        self.chunks.clear()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalChunk]:
        """
        Search chunks using a simple keyword overlap score.

        This is a lightweight baseline that can later be replaced with
        embedding-based retrieval.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored: list[RetrievalChunk] = []

        for chunk in self.chunks:
            score = self._score_chunk(chunk.text, chunk.file_path, query_terms)
            if score > 0:
                scored.append(
                    RetrievalChunk(
                        text=chunk.text,
                        file_path=chunk.file_path,
                        chunk_id=chunk.chunk_id,
                        language=chunk.language,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        score=score,
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _score_chunk(self, text: str, file_path: str, query_terms: set[str]) -> float:
        text_terms = self._tokenize(text)
        path_terms = self._tokenize(file_path)

        text_matches = len(query_terms & text_terms)
        path_matches = len(query_terms & path_terms)

        # Slightly favor matches in content over file path.
        return float(text_matches * 2 + path_matches)

    def _tokenize(self, text: str) -> set[str]:
        tokens: set[str] = set()
        current = []

        for char in text.lower():
            if char.isalnum():
                current.append(char)
            else:
                if current:
                    tokens.add("".join(current))
                    current = []

        if current:
            tokens.add("".join(current))

        return tokens