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


@dataclass(frozen=True)
class RetrievalPolicy:
    """
    Adaptive retrieval policy controls.

    max_chunks_cap:
        Absolute upper bound on retrieved chunks.
    max_context_chars:
        Safety limit for the total retrieved chunk text.
    max_chunks_per_file:
        Prevents a single file from dominating the prompt.
    small_repo_file_threshold:
        File count threshold for small repos.
    large_repo_file_threshold:
        File count threshold for large repos.
    coverage_ratio:
        Fraction of the budget reserved for coverage-first selection.
    relevance_ratio:
        Fraction of the budget reserved for relevance fill.
    """

    max_chunks_cap: int = 20
    max_context_chars: int = 24_000
    max_chunks_per_file: int = 2
    small_repo_file_threshold: int = 20
    large_repo_file_threshold: int = 100
    coverage_ratio: float = 0.6
    relevance_ratio: float = 0.4


class Retriever:
    """
    High-level retrieval interface for RAG queries.
    """

    def __init__(self, index: ChunkIndex, policy: RetrievalPolicy | None = None) -> None:
        self.index = index
        self.policy = policy or RetrievalPolicy()

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Adaptive hybrid retrieval.

        This method:
        1. Scores all chunks using the index search scoring.
        2. Groups chunks by file.
        3. Selects one best chunk per file first, when feasible.
        4. Fills remaining slots with globally relevant chunks.
        5. Enforces chunk-count and character-budget caps.
        """
        scored_chunks = self.index.search(query=query, top_k=max(top_k, self.policy.max_chunks_cap))

        if not scored_chunks:
            return RetrievalResult(query=query, chunks=[])

        file_count = len({chunk.file_path for chunk in self.index.chunks})
        target_k = self._choose_target_k(file_count=file_count, requested_k=top_k)

        selected = self._adaptive_select(
            scored_chunks=scored_chunks,
            target_k=target_k,
            file_count=file_count,
        )

        return RetrievalResult(query=query, chunks=selected)

    def _choose_target_k(self, file_count: int, requested_k: int) -> int:
        """
        Decide how many chunks to return based on repo size and caller preference.
        """
        cap = min(self.policy.max_chunks_cap, max(1, requested_k))

        if file_count <= self.policy.small_repo_file_threshold:
            # Small repos: favor broad coverage.
            return min(file_count, cap)

        if file_count <= self.policy.large_repo_file_threshold:
            # Medium repos: cover broadly, then add some relevance slots.
            coverage_budget = max(1, int(cap * self.policy.coverage_ratio))
            relevance_budget = max(0, cap - coverage_budget)
            return min(cap, coverage_budget + relevance_budget)

        # Large repos: hard cap dominates.
        return cap

    def _adaptive_select(
        self,
        scored_chunks: list[RetrievalChunk],
        target_k: int,
        file_count: int,
    ) -> list[RetrievalChunk]:
        """
        Hybrid coverage + relevance selection with per-file and context limits.
        """
        if target_k <= 0:
            return []

        chunks_by_file: dict[str, list[RetrievalChunk]] = {}
        for chunk in scored_chunks:
            chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

        for file_chunks in chunks_by_file.values():
            file_chunks.sort(key=lambda c: c.score, reverse=True)

        ranked_files = sorted(
            chunks_by_file.items(),
            key=lambda item: item[1][0].score if item[1] else 0.0,
            reverse=True,
        )

        selected: list[RetrievalChunk] = []
        selected_ids: set[str] = set()
        selected_counts_by_file: dict[str, int] = {}
        total_chars = 0

        # Phase 1: coverage-first selection.
        coverage_budget = self._coverage_budget(target_k=target_k, file_count=file_count)
        for file_path, file_chunks in ranked_files:
            if len(selected) >= coverage_budget or len(selected) >= target_k:
                break

            best_chunk = file_chunks[0]
            if self._can_add_chunk(
                chunk=best_chunk,
                selected_ids=selected_ids,
                selected_counts_by_file=selected_counts_by_file,
                total_chars=total_chars,
            ):
                selected.append(best_chunk)
                selected_ids.add(best_chunk.chunk_id)
                selected_counts_by_file[file_path] = selected_counts_by_file.get(file_path, 0) + 1
                total_chars += len(best_chunk.text)

        # Phase 2: relevance fill.
        remaining_chunks = sorted(scored_chunks, key=lambda c: c.score, reverse=True)
        for chunk in remaining_chunks:
            if len(selected) >= target_k:
                break

            if chunk.chunk_id in selected_ids:
                continue

            if not self._can_add_chunk(
                chunk=chunk,
                selected_ids=selected_ids,
                selected_counts_by_file=selected_counts_by_file,
                total_chars=total_chars,
            ):
                continue

            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            selected_counts_by_file[chunk.file_path] = selected_counts_by_file.get(chunk.file_path, 0) + 1
            total_chars += len(chunk.text)

        return selected

    def _coverage_budget(self, target_k: int, file_count: int) -> int:
        """
        How many slots to reserve for broad file coverage.
        """
        if file_count <= self.policy.small_repo_file_threshold:
            return min(target_k, file_count)

        if file_count <= self.policy.large_repo_file_threshold:
            return max(1, int(target_k * self.policy.coverage_ratio))

        # Very large repos: keep some coverage but preserve the hard cap.
        return max(1, min(file_count, int(target_k * self.policy.coverage_ratio)))

    def _can_add_chunk(
        self,
        chunk: RetrievalChunk,
        selected_ids: set[str],
        selected_counts_by_file: dict[str, int],
        total_chars: int,
    ) -> bool:
        """
        Check hard constraints before adding a chunk.
        """
        if chunk.chunk_id in selected_ids:
            return False

        if selected_counts_by_file.get(chunk.file_path, 0) >= self.policy.max_chunks_per_file:
            return False

        if total_chars + len(chunk.text) > self.policy.max_context_chars:
            return False

        return True