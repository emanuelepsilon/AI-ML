import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "gemini-3.5-flash"
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def extract_interaction_answer(data: dict) -> tuple[str, list[dict]]:
    text_blocks = []
    citations_by_url = {}

    for step in data.get("steps", []):
        if step.get("type") != "model_output":
            continue

        for block in step.get("content", []):
            if block.get("type") != "text":
                continue

            text = block.get("text", "")
            if text:
                text_blocks.append(text)

            for annotation in block.get("annotations", []) or []:
                if annotation.get("type") != "url_citation":
                    continue

                url = annotation.get("url")
                if not url:
                    continue

                citations_by_url[url] = {
                    "title": annotation.get("title") or url,
                    "url": url,
                }

    return "\n".join(text_blocks).strip(), list(citations_by_url.values())


def ask_gemini(prompt: str) -> tuple[str, list[dict]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Create web_agent/.env and add "
            "GEMINI_API_KEY=your_key_here"
        )

    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    payload = {
        "model": model,
        "input": (
            "You are a practical assistant with access to Google Search. "
            "Use search whenever the user's question depends on recent, "
            "changing, niche, or source-sensitive information. If you use "
            "search, ground your answer in the results and keep the answer "
            "concise.\n\n"
            f"User question: {prompt}"
        ),
        "tools": [{"type": "google_search"}],
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        if error.code == 429:
            raise RuntimeError(
                "Gemini refused this request because the current API quota was exceeded. "
                "This can happen quickly with Google Search grounding. Check "
                "https://ai.dev/rate-limit or try again after your quota resets."
            ) from error
        raise RuntimeError(f"Gemini API error {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Gemini API: {error.reason}") from error

    answer, citations = extract_interaction_answer(data)
    if not answer:
        raise RuntimeError(f"Gemini returned no answer: {json.dumps(data, indent=2)}")

    return answer, citations


def print_answer(prompt: str) -> bool:
    try:
        answer, citations = ask_gemini(prompt)
    except RuntimeError as error:
        print(error)
        return False

    print()
    print(f"Agent: {answer}")
    if citations:
        print()
        print("Sources:")
        for index, citation in enumerate(citations, start=1):
            print(f"{index}. {citation['title']} - {citation['url']}")
    print()
    return True


def chat_loop() -> int:
    print("Terminal Web Agent")
    print("Type /exit to quit.")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Bye.")
            return 0

        if not prompt:
            continue

        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Bye.")
            return 0

        print_answer(prompt)


def main() -> int:
    load_env_file(Path(".env"))

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
        if not prompt:
            print("No question given.")
            return 1
        return 0 if print_answer(prompt) else 1

    return chat_loop()


if __name__ == "__main__":
    raise SystemExit(main())
