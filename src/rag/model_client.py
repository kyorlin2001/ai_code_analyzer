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
    raw_response: dict[str, Any] | None = None


class ModelClient:
    """
    Client for a hosted chat-completion style model API.
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

        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
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

    def _extract_text(self, data: dict[str, Any]) -> str:
        """
        Extract text from a chat-completion style response.

        Supports responses shaped like:
        - {"choices": [{"message": {"content": "..."}}]}
        - {"output_text": "..."}
        - {"text": "..."}
        """
        if "choices" in data and data["choices"]:
            first_choice = data["choices"][0]
            message = first_choice.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        for key in ("output_text", "text", "content"):
            value = data.get(key)
            if isinstance(value, str):
                return value.strip()

        raise ValueError("Could not extract model text from response.")