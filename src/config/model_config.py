from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for the hosted model and RAG behavior.
    """

    provider_name: str = "huggingface"
    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    api_key: str | None = None

    temperature: float = 0.2
    max_tokens: int = 1200
    top_k: int = 5
    max_context_chars: int = 24_000

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """
        Load config from environment variables.

        Supported environment variables:
        - MODEL_PROVIDER_NAME
        - MODEL_NAME
        - HF_TOKEN
        - MODEL_TEMPERATURE
        - MODEL_MAX_TOKENS
        - RAG_TOP_K
        - RAG_MAX_CONTEXT_CHARS
        """
        return cls(
            provider_name=os.getenv("MODEL_PROVIDER_NAME", "huggingface"),
            model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct"),
            api_key=os.getenv("HF_TOKEN"),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "1200")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "24000")),
        )