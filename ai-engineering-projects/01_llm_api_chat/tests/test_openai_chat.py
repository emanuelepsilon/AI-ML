from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_chat.config import ChatConfig
from llm_chat.openai_chat import OpenAIChatSession


class FakeResponses:
    def __init__(self, output_text: str = "Test answer"):
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text: str = "Test answer"):
        self.responses = FakeResponses(output_text=output_text)


def make_config(max_history_messages: int = 12) -> ChatConfig:
    return ChatConfig(
        model="test-model",
        temperature=None,
        max_history_messages=max_history_messages,
        system_prompt="Answer like a careful engineer.",
        log_level="INFO",
    )


def test_ask_sends_developer_prompt_and_user_message():
    client = FakeClient(output_text="RAG retrieves context before generation.")
    session = OpenAIChatSession(client=client, config=make_config())

    answer = session.ask("What is RAG?")

    assert answer == "RAG retrieves context before generation."
    request = client.responses.calls[0]
    assert request["model"] == "test-model"
    assert request["input"][0] == {
        "role": "developer",
        "content": "Answer like a careful engineer.",
    }
    assert request["input"][-1] == {"role": "user", "content": "What is RAG?"}


def test_ask_stores_user_and_assistant_history():
    client = FakeClient(output_text="Hello.")
    session = OpenAIChatSession(client=client, config=make_config())

    session.ask("Hi")

    assert session.history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ]


def test_ask_rejects_empty_messages():
    session = OpenAIChatSession(client=FakeClient(), config=make_config())

    with pytest.raises(ValueError, match="Message cannot be empty"):
        session.ask("   ")


def test_ask_rejects_empty_model_output():
    session = OpenAIChatSession(client=FakeClient(output_text="   "), config=make_config())

    with pytest.raises(RuntimeError, match="returned no text"):
        session.ask("Say something")


def test_history_is_trimmed_to_configured_limit():
    session = OpenAIChatSession(client=FakeClient(output_text="ok"), config=make_config(max_history_messages=2))

    session.ask("one")
    session.ask("two")

    assert session.history == [
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "ok"},
    ]

