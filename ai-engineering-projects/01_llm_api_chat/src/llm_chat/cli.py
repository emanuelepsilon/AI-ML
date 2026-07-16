from __future__ import annotations

import logging

from llm_chat.config import has_api_key, load_config
from llm_chat.openai_chat import OpenAIChatSession, create_openai_client


EXIT_COMMANDS = {"exit", "quit", ":q"}


def main() -> int:
    try:
        config = load_config()
    except ValueError as exc:
        print(f"config error> {exc}")
        return 2

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    if not has_api_key():
        print("config error> The OPENAI_API_KEY environment variable is missing.")
        print("config error> Copy .env.example to .env and add your API key.")
        return 2

    try:
        session = OpenAIChatSession(client=create_openai_client(), config=config)
    except RuntimeError as exc:
        print(f"setup error> {exc}")
        return 2

    print(f"LLM API Chat using model {config.model}. Type exit, quit or :q to stop.")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue

        if user_text.lower() in EXIT_COMMANDS:
            return 0

        try:
            answer = session.ask(user_text)
        except Exception as exc:
            logging.exception("Chat request failed")
            print(f"error> {exc}")
            continue

        print(f"assistant> {answer}\n")


if __name__ == "__main__":
    raise SystemExit(main())

