from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise AI engineering study assistant. Explain practical concepts "
    "clearly, prefer runnable examples, and call out tradeoffs."
)


@dataclass(frozen=True)
class ChatConfig:
    model: str
    temperature: float | None
    max_history_messages: int
    system_prompt: str
    log_level: str


def load_config() -> ChatConfig:
    load_dotenv()

    return ChatConfig(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        temperature=_optional_float("OPENAI_TEMPERATURE"),
        max_history_messages=max(
            2,
            _optional_int("LLM_CHAT_MAX_HISTORY_MESSAGES", default=12),
        ),
        system_prompt=os.getenv("LLM_CHAT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        log_level=os.getenv("LLM_CHAT_LOG_LEVEL", "INFO"),
    )


def has_api_key() -> bool:
    load_dotenv()
    return bool(os.getenv("OPENAI_API_KEY"))


def _optional_float(name: str) -> float | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}.") from exc


def _optional_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc

