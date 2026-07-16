# Project 1: LLM API Chat CLI

This is the first portfolio project in the AI engineering internship roadmap. It is a small Python chat app that calls a hosted Large Language Model through an API.

The point is not to build a fancy chatbot. The point is to learn the real engineering shape: configuration, secrets, API calls, conversation state, error handling, tests and documentation.

## What It Solves

You can ask questions from a terminal and receive model responses while keeping a short conversation history. This is the base pattern behind RAG assistants, agents and applied AI prototypes.

## Technical Stack

- Python 3.11+
- OpenAI Python SDK
- OpenAI Responses API
- python-dotenv
- pytest

## Folder Structure

```text
01_llm_api_chat/
  README.md
  pyproject.toml
  .env.example
  src/llm_chat/
    __init__.py
    cli.py
    config.py
    openai_chat.py
  tests/
    test_openai_chat.py
```

## Install

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
copy .env.example .env
notepad .env
```

Put your real API key in `.env`.

This activation-free style avoids PowerShell execution policy issues. You can still activate the environment manually if your system allows it.

## Run

```powershell
.\.venv\Scripts\python.exe -m llm_chat.cli
```

Then type a message:

```text
you> Explain RAG in three sentences.
assistant> ...
```

Type `exit`, `quit` or `:q` to stop.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The tests use a fake client, so they do not call the API and do not spend money.

## Files To Understand

`src/llm_chat/config.py` loads environment variables. This teaches configuration, API key hygiene and model selection.

`src/llm_chat/openai_chat.py` owns the model call. This teaches how a Python system wraps an LLM API instead of scattering API calls everywhere.

`src/llm_chat/cli.py` owns the terminal interface. This teaches separation of concerns: the user interface is separate from the model integration.

`tests/test_openai_chat.py` tests behavior with a fake client. This teaches how real AI systems test orchestration without depending on external APIs.

## Learning While Building

After writing the config layer, you learned that production AI apps should not hardcode secrets, model names or settings. Companies ask for this because interns often work with shared API keys and different environments.

After writing the chat client, you learned that an LLM call is just one step in an application pipeline. The application controls instructions, input messages, output handling and failure behavior.

After writing the CLI, you learned that a simple interface can still be professional if the business logic is separated from the presentation layer.

After writing the tests, you learned that AI systems can be tested even when the model is probabilistic. You test the deterministic parts: message construction, history handling and error behavior.

## Common Errors

Missing API key:

```text
The OPENAI_API_KEY environment variable is missing.
```

Fix: copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

Model not available:

```text
The model does not exist or you do not have access.
```

Fix: set `OPENAI_MODEL` in `.env` to a model available in your account.

PowerShell activation blocked:

```text
running scripts is disabled on this system
```

Fix: use the activation-free commands in this README.

Package missing:

```text
No module named openai
```

Fix: run `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

Rate limit or quota error:

Fix: wait, lower usage, check billing or use a cheaper model.

## README Notes For Your Portfolio

Include these points when presenting the project:

- Built a Python CLI chat app using a hosted LLM API.
- Used environment variables for API keys and model configuration.
- Implemented conversation history with a configurable limit.
- Added error handling for missing dependencies, failed API calls and empty model responses.
- Added tests with a fake API client to avoid network calls and API cost.

## Interview Notes

You should be able to explain:

- What an LLM API is.
- Why API keys belong in environment variables.
- How the application builds messages before calling the model.
- Why tests should mock external APIs.
- Why this project is the base for RAG and agents.

## Extensions

- Add streaming responses.
- Save conversations to JSON.
- Add a FastAPI endpoint.
- Track estimated token usage and cost.
- Add a simple prompt evaluation dataset.

