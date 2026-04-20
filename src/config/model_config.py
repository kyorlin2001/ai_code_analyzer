from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for the hosted model and RAG behavior.
    """

    provider_name: str = "huggingface"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: str | None = None

    temperature: float = 0.2
    max_tokens: int = 1200
    top_k: int = 5
    max_context_chars: int = 24_000

    # Retrieval policy controls
    retrieval_max_chunks_cap: int = 20
    retrieval_max_chunks_per_file: int = 2
    retrieval_small_repo_file_threshold: int = 20
    retrieval_large_repo_file_threshold: int = 100
    retrieval_coverage_ratio: float = 0.6

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """
        Load config from environment variables.

        Supported environment variables:
        - MODEL_PROVIDER_NAME
        - MODEL_NAME
        - MODEL_API_KEY
        - MODEL_TEMPERATURE
        - MODEL_MAX_TOKENS
        - RAG_TOP_K
        - RAG_MAX_CONTEXT_CHARS
        - RAG_RETRIEVAL_MAX_CHUNKS_CAP
        - RAG_RETRIEVAL_MAX_CHUNKS_PER_FILE
        - RAG_RETRIEVAL_SMALL_REPO_FILE_THRESHOLD
        - RAG_RETRIEVAL_LARGE_REPO_FILE_THRESHOLD
        - RAG_RETRIEVAL_COVERAGE_RATIO
        """
        return cls(
            provider_name=os.getenv("MODEL_PROVIDER_NAME", "together"),
            model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct"),
            api_key=os.getenv("MODEL_API_KEY"),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "1200")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "24000")),
            retrieval_max_chunks_cap=int(os.getenv("RAG_RETRIEVAL_MAX_CHUNKS_CAP", "20")),
            retrieval_max_chunks_per_file=int(os.getenv("RAG_RETRIEVAL_MAX_CHUNKS_PER_FILE", "2")),
            retrieval_small_repo_file_threshold=int(
                os.getenv("RAG_RETRIEVAL_SMALL_REPO_FILE_THRESHOLD", "20")
            ),
            retrieval_large_repo_file_threshold=int(
                os.getenv("RAG_RETRIEVAL_LARGE_REPO_FILE_THRESHOLD", "100")
            ),
            retrieval_coverage_ratio=float(os.getenv("RAG_RETRIEVAL_COVERAGE_RATIO", "0.6")),
        )