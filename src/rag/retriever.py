from __future__ import annotations

from dataclasses import dataclass, field

from models.retrieval_chunk import RetrievalChunk
from rag.index import ChunkIndex


@dataclass
class RetrievalResult:
    """
    Result of a repository retrieval query.
    """

    query: str
    chunks: list[RetrievalChunk] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Retriever:
    """
    High-level retrieval interface for RAG queries.
    """

    def __init__(self, index: ChunkIndex) -> None:
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        chunks = self.index.search(query=query, top_k=top_k)
        return RetrievalResult(query=query, chunks=chunks)