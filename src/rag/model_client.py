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
    Client for Hugging Face InferenceClient using chat_completion.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()

        if not self.config.api_key:
            raise ValueError("HF_TOKEN is not configured.")
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
        prompt_txt = self._build_prompt(prompt)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt_txt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
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
        print("HF RESPONSE TYPE:", type(response))
        print("HF RESPONSE REPR:", repr(response))

        choices = getattr(response, "choices", None)
        print("HF CHOICES:", choices)

        if choices and len(choices) > 0:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            print("HF MESSAGE:", message)

            if message is not None:
                content = getattr(message, "content", None)
                print("HF CONTENT:", content)
                if isinstance(content, str):
                    return content.strip()

        if isinstance(response, str):
            return response.strip()

        raise ValueError(f"Could not extract model text from Hugging Face response: {type(response)}")