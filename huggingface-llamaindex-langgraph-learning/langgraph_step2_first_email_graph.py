"""LangGraph email-triage workflow with model-based routing."""

from typing import Any, Literal, Optional

import requests
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2:7b"


def call_ollama(prompt: str) -> str:
    """
    Call local Ollama and return text.

    This keeps the example dependency-light:
    no OpenAI key, no LangChain model wrapper, just a local HTTP call.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


class EmailState(TypedDict):
    email: dict[str, Any]
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_category: Optional[str]
    email_draft: Optional[str]
    notes: list[str]


def read_email(state: EmailState) -> dict:
    """Read/log the incoming email."""
    email = state["email"]

    print("--- read_email node ---")
    print(f"From: {email['sender']}")
    print(f"Subject: {email['subject']}")

    return {
        "notes": state["notes"]
        + [f"Read email from {email['sender']} about {email['subject']}."]
    }


def classify_email(state: EmailState) -> dict:
    """
    Use an LLM to classify the email.

    To make parsing simple, we ask the model to answer with only a tiny format:

        spam | reason | category

    Example:
        yes | asks for bank details and processing fee | scam
        no | legitimate service inquiry | inquiry
    """
    email = state["email"]

    print("--- classify_email node ---")

    prompt = f"""
You are an email classifier.

Classify this email as spam or legitimate.
Return ONLY this exact format:
spam: yes/no
reason: short reason
category: one of inquiry, complaint, thank_you, request, scam, unknown

Email:
From: {email["sender"]}
Subject: {email["subject"]}
Body: {email["body"]}
"""

    llm_response = call_ollama(prompt)
    print("LLM classification response:")
    print(llm_response)

    lowered = llm_response.lower()
    is_spam = "spam: yes" in lowered

    reason = "No reason parsed."
    category = "unknown"

    for line in llm_response.splitlines():
        normalized_line = line.strip()
        lowered_line = normalized_line.lower()

        if lowered_line.startswith("reason:"):
            reason = normalized_line.split(":", 1)[1].strip()

        if lowered_line.startswith("category:"):
            category = normalized_line.split(":", 1)[1].strip()

    return {
        "is_spam": is_spam,
        "spam_reason": reason if is_spam else None,
        "email_category": category,
        "notes": state["notes"] + [f"Classified email. Spam={is_spam}."],
    }


def handle_spam(state: EmailState) -> dict:
    """Handle the spam path."""
    print("--- handle_spam node ---")
    print(f"Spam reason: {state['spam_reason']}")

    return {
        "notes": state["notes"]
        + [f"Moved email to spam. Reason: {state['spam_reason']}."]
    }


def draft_response(state: EmailState) -> dict:
    """Use an LLM to draft a response for legitimate emails."""
    email = state["email"]

    print("--- draft_response node ---")

    prompt = f"""
You are a polite professional assistant.

Draft a short preliminary reply to this legitimate email.
Do not promise anything final.
Keep it concise.

Email:
From: {email["sender"]}
Subject: {email["subject"]}
Body: {email["body"]}

Category: {state["email_category"]}
"""

    draft = call_ollama(prompt)

    return {
        "email_draft": draft,
        "notes": state["notes"] + ["Drafted a preliminary response."],
    }


def notify_user(state: EmailState) -> dict:
    """Print the final legitimate-email summary."""
    email = state["email"]

    print("--- notify_user node ---")
    print("\nEmail is legitimate.")
    print(f"From: {email['sender']}")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nDraft response:")
    print(state["email_draft"])

    return {"notes": state["notes"] + ["Presented draft response for review."]}


def route_email(state: EmailState) -> Literal["spam", "legitimate"]:
    print("--- route_email conditional edge ---")

    if state["is_spam"]:
        return "spam"

    return "legitimate"


builder = StateGraph(EmailState)

builder.add_node("read_email", read_email)
builder.add_node("classify_email", classify_email)
builder.add_node("handle_spam", handle_spam)
builder.add_node("draft_response", draft_response)
builder.add_node("notify_user", notify_user)

builder.add_edge(START, "read_email")
builder.add_edge("read_email", "classify_email")

builder.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response",
    },
)

builder.add_edge("handle_spam", END)
builder.add_edge("draft_response", "notify_user")
builder.add_edge("notify_user", END)

graph = builder.compile()

LEGITIMATE_EMAIL = {
    "sender": "customer@example.com",
    "subject": "Question about mortgage requirements",
    "body": (
        "Hello, I would like to understand the minimum credit score and "
        "debt-to-income expectations for a mortgage. Could someone explain?"
    ),
}

SPAM_EMAIL = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": (
        "Congratulations! Send your bank details and a processing fee of $100 "
        "to claim your prize immediately."
    ),
}


def initial_state(email: dict[str, Any]) -> EmailState:
    return {
        "email": email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "notes": [],
    }


def run_email_example(label: str, email: dict[str, Any]):
    print(f"\n=== Processing {label} email ===")

    try:
        final_state = graph.invoke(initial_state(email))
    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama.")
        print("Start Ollama in another terminal, then run this file again:")
        print("    ollama serve")
        return

    print("\nFinal state:")
    print(final_state)


if __name__ == "__main__":
    run_email_example("legitimate", LEGITIMATE_EMAIL)
    run_email_example("spam", SPAM_EMAIL)
