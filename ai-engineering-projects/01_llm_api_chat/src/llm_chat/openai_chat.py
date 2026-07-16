from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_chat.config import ChatConfig


Message = dict[str, str]


def create_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `python -m pip install -e \".[dev]\"`."
        ) from exc

    return OpenAI()


@dataclass
class OpenAIChatSession:
    client: Any
    config: ChatConfig
    history: list[Message] = field(default_factory=list)

    def ask(self, user_text: str) -> str:
        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("Message cannot be empty.")

        messages = self._build_messages(clean_text)
        request = self._build_request(messages)
        response = self.client.responses.create(**request)
        answer = getattr(response, "output_text", "").strip()

        if not answer:
            raise RuntimeError("The model returned no text.")

        self.history.append({"role": "user", "content": clean_text})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return answer

    def _build_messages(self, user_text: str) -> list[Message]:
        return [
            {"role": "developer", "content": self.config.system_prompt},
            *self.history,
            {"role": "user", "content": user_text},
        ]

    def _build_request(self, messages: list[Message]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "input": messages,
        }

        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature

        return request

    def _trim_history(self) -> None:
        extra_messages = len(self.history) - self.config.max_history_messages
        if extra_messages > 0:
            del self.history[:extra_messages]

