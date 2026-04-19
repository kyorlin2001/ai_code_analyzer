from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smolagents import CodeAgent, InferenceClientModel

from config.model_config import ModelConfig
from rag.prompt_builder import PromptBundle


@dataclass
class ModelResponse:
    """
    Raw response payload returned by the hosted model client.
    """

    text: str
    raw_response: Any | None = None


class ModelClient:
    """
    Client for Hugging Face smolagents HfApiModel.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()

        if not self.config.api_key:
            raise ValueError("HF_TOKEN is not configured.")
        if not self.config.model_name:
            raise ValueError("MODEL_NAME is not configured.")

        self.model = InferenceClientModel(
            model_id=self.config.model_name,
            token=self.config.api_key,
        )

    def complete(self, prompt: PromptBundle) -> ModelResponse:
        """
        Send a prompt to the hosted model and return the model's response text.
        """
        full_prompt = self._build_prompt(prompt)

        try:
            response = self.model(full_prompt)
        except Exception as exc:
            raise RuntimeError(f"Hugging Face model request failed: {exc}") from exc

        text = self._extract_text(response)
        return ModelResponse(text=text, raw_response=response)

    def _build_prompt(self, prompt: PromptBundle) -> str:
        return (
            f"{prompt.system_prompt}\n\n"
            f"{prompt.user_prompt}"
        )

    def _extract_text(self, response: Any) -> str:
        """
        Normalize smolagents/HF model outputs.
        """
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, dict):
            for key in ("generated_text", "text", "content", "answer"):
                value = response.get(key)
                if isinstance(value, str):
                    return value.strip()

        if hasattr(response, "content") and isinstance(response.content, str):
            return response.content.strip()

        return str(response).strip()