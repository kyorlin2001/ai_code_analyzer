from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from huggingface_hub import InferenceClient

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
    Client for Hugging Face InferenceClient.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()

        if not self.config.api_key:
            raise ValueError("MODEL_API_KEY is not configured.")
        if not self.config.model_name:
            raise ValueError("MODEL_NAME is not configured.")

        self.client = InferenceClient(
            model=self.config.model_name,
            token=self.config.api_key,
        )

    def complete(self, prompt: PromptBundle) -> ModelResponse:
        """
        Send a prompt to the hosted model and return the model's response text.
        """
        full_prompt = self._build_prompt(prompt)

        try:
            response = self.client.text_generation(
                full_prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                return_full_text=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Hugging Face inference request failed: {exc}") from exc

        text = self._extract_text(response)
        return ModelResponse(text=text, raw_response=response)

    def _build_prompt(self, prompt: PromptBundle) -> str:
        return (
            f"{prompt.system_prompt}\n\n"
            f"{prompt.user_prompt}"
        )

    def _extract_text(self, response: Any) -> str:
        """
        Normalize HF text generation outputs.
        """
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                for key in ("generated_text", "text", "content"):
                    value = first.get(key)
                    if isinstance(value, str):
                        return value.strip()

        if isinstance(response, dict):
            for key in ("generated_text", "text", "content"):
                value = response.get(key)
                if isinstance(value, str):
                    return value.strip()

        raise ValueError("Could not extract model text from Hugging Face response.")