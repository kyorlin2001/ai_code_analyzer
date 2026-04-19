from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from config.model_config import ModelConfig
from rag.prompt_builder import PromptBundle


@dataclass
class ModelResponse:
    """
    Raw response payload returned by the hosted model client.
    """

    text: str
    raw_response: dict[str, Any] | list[Any] | None = None


class ModelClient:
    """
    Client for the Hugging Face Serverless Inference API.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()

    def complete(self, prompt: PromptBundle) -> ModelResponse:
        """
        Send a prompt to the hosted model and return the model's response text.
        """
        if not self.config.api_base_url:
            raise ValueError("MODEL_API_BASE_URL is not configured.")
        if not self.config.api_key:
            raise ValueError("MODEL_API_KEY is not configured.")

        full_prompt = (
            f"{prompt.system_prompt}\n\n"
            f"{prompt.user_prompt}"
        )

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens,
                "return_full_text": False,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.config.api_base_url,
            json=payload,
            headers=headers,
            timeout=90,
        )
        response.raise_for_status()

        data = response.json()
        text = self._extract_text(data)

        return ModelResponse(text=text, raw_response=data)

    def _extract_text(self, data: dict[str, Any] | list[Any]) -> str:
        """
        Extract text from Hugging Face inference responses.

        Common shapes:
        - [{"generated_text": "..."}]
        - {"generated_text": "..."}
        - {"text": "..."}
        """
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for key in ("generated_text", "text", "content"):
                    value = first.get(key)
                    if isinstance(value, str):
                        return value.strip()

        if isinstance(data, dict):
            for key in ("generated_text", "text", "content"):
                value = data.get(key)
                if isinstance(value, str):
                    return value.strip()

        raise ValueError("Could not extract model text from Hugging Face response.")