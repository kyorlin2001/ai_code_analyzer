import unittest

from models.retrieval_chunk import RetrievalChunk
from rag.retriever import RetrievalPolicy, Retriever


class FakeChunkIndex:
    def __init__(self, chunks: list[RetrievalChunk]) -> None:
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[RetrievalChunk]:
        return self.chunks[:top_k]


class RetrieverSanityTests(unittest.TestCase):
    def test_adaptive_retrieval_balances_coverage_and_limits(self) -> None:
        chunks = [
            RetrievalChunk(
                text="alpha function query match",
                file_path="src/a.py",
                chunk_id="src/a.py::chunk-0",
                score=10.0,
            ),
            RetrievalChunk(
                text="alpha helper query match",
                file_path="src/a.py",
                chunk_id="src/a.py::chunk-1",
                score=9.0,
            ),
            RetrievalChunk(
                text="beta function query match",
                file_path="src/b.py",
                chunk_id="src/b.py::chunk-0",
                score=8.0,
            ),
            RetrievalChunk(
                text="gamma function query match",
                file_path="src/c.py",
                chunk_id="src/c.py::chunk-0",
                score=7.0,
            ),
            RetrievalChunk(
                text="delta function query match",
                file_path="src/d.py",
                chunk_id="src/d.py::chunk-0",
                score=6.0,
            ),
        ]

        index = FakeChunkIndex(chunks)
        retriever = Retriever(
            index=index,
            policy=RetrievalPolicy(
                max_chunks_cap=3,
                max_context_chars=10_000,
                max_chunks_per_file=1,
                small_repo_file_threshold=20,
                large_repo_file_threshold=100,
                coverage_ratio=0.6,
            ),
        )

        result = retriever.retrieve(query="query", top_k=10)

        self.assertLessEqual(len(result.chunks), 3)
        self.assertEqual(len({chunk.file_path for chunk in result.chunks}), len(result.chunks))
        self.assertIn("src/a.py", {chunk.file_path for chunk in result.chunks})
        self.assertIn("src/b.py", {chunk.file_path for chunk in result.chunks})

    def test_context_budget_blocks_overflow(self) -> None:
        chunks = [
            RetrievalChunk(
                text="x" * 60,
                file_path="src/a.py",
                chunk_id="src/a.py::chunk-0",
                score=10.0,
            ),
            RetrievalChunk(
                text="y" * 60,
                file_path="src/b.py",
                chunk_id="src/b.py::chunk-0",
                score=9.0,
            ),
        ]

        index = FakeChunkIndex(chunks)
        retriever = Retriever(
            index=index,
            policy=RetrievalPolicy(
                max_chunks_cap=5,
                max_context_chars=80,
                max_chunks_per_file=1,
                small_repo_file_threshold=20,
                large_repo_file_threshold=100,
                coverage_ratio=0.6,
            ),
        )

        result = retriever.retrieve(query="query", top_k=5)

        self.assertLessEqual(sum(len(chunk.text) for chunk in result.chunks), 80)
        self.assertLessEqual(len(result.chunks), 2)

    def test_retriever_uses_default_policy_when_not_configured(self) -> None:
        chunks = [
            RetrievalChunk(
                text="query match one",
                file_path="src/a.py",
                chunk_id="src/a.py::chunk-0",
                score=3.0,
            ),
            RetrievalChunk(
                text="query match two",
                file_path="src/b.py",
                chunk_id="src/b.py::chunk-0",
                score=2.0,
            ),
            RetrievalChunk(
                text="query match three",
                file_path="src/c.py",
                chunk_id="src/c.py::chunk-0",
                score=1.0,
            ),
        ]

        index = FakeChunkIndex(chunks)
        retriever = Retriever(index=index)

        result = retriever.retrieve(query="query", top_k=5)

        self.assertGreater(len(result.chunks), 0)
        self.assertLessEqual(len(result.chunks), 5)
        self.assertEqual(result.query, "query")


if __name__ == "__main__":
    unittest.main()