from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

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
    Client for Hugging Face's OpenAI-compatible router API.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()

        if not self.config.api_base_url:
            raise ValueError("MODEL_API_BASE_URL is not configured.")
        if not self.config.api_key:
            raise ValueError("MODEL_API_KEY is not configured.")

        self.client = OpenAI(
            base_url=self.config.api_base_url,
            api_key=self.config.api_key,
        )

    def complete(self, prompt: PromptBundle) -> ModelResponse:
        """
        Send a prompt to the hosted model and return the model's response text.
        """
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
        )

        text = self._extract_text(response)
        return ModelResponse(text=text, raw_response=response)

    def _extract_text(self, response: Any) -> str:
        """
        Extract text from an OpenAI-compatible chat completion response.
        """
        try:
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
        except Exception:
            pass

        raise ValueError("Could not extract model text from response.")