"""LangGraph state, node, edge, and conditional-routing example."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class BankState(TypedDict):
    customer_name: str
    credit_score: int
    debt_to_income: float
    risk_level: str
    message: str


def load_customer(state: BankState) -> dict:
    print("--- load_customer node ---")

    customer_name = state["customer_name"].lower().strip()

    if customer_name == "alice":
        return {
            "credit_score": 720,
            "debt_to_income": 0.31,
            "message": "Loaded Alice Rivera.",
        }

    if customer_name == "ben":
        return {
            "credit_score": 610,
            "debt_to_income": 0.47,
            "message": "Loaded Ben Carter.",
        }

    return {
        "credit_score": 0,
        "debt_to_income": 1.0,
        "message": f"No known customer named {state['customer_name']!r}.",
    }


def assess_risk(state: BankState) -> dict:
    print("--- assess_risk node ---")

    risk_flags = []

    if state["credit_score"] < 640:
        risk_flags.append("low credit score")

    if state["debt_to_income"] > 0.40:
        risk_flags.append("high debt-to-income")

    if risk_flags:
        return {
            "risk_level": "high",
            "message": state["message"] + " Risk flags: " + ", ".join(risk_flags),
        }

    return {
        "risk_level": "low",
        "message": state["message"] + " No simple risk flags found.",
    }


def low_risk_summary(state: BankState) -> dict:
    print("--- low_risk_summary node ---")
    return {
        "message": (
            state["message"]
            + " Route: low-risk summary. Final decision still requires human review."
        )
    }


def high_risk_summary(state: BankState) -> dict:
    print("--- high_risk_summary node ---")
    return {
        "message": (
            state["message"] + " Route: high-risk summary. Escalate to human review."
        )
    }


def choose_summary_path(
    state: BankState,
) -> Literal[
    "low_risk_summary",
    "high_risk_summary",
]:
    print("--- choose_summary_path edge function ---")

    if state["risk_level"] == "high":
        return "high_risk_summary"

    return "low_risk_summary"


builder = StateGraph(BankState)

builder.add_node("load_customer", load_customer)
builder.add_node("assess_risk", assess_risk)
builder.add_node("low_risk_summary", low_risk_summary)
builder.add_node("high_risk_summary", high_risk_summary)

builder.add_edge(START, "load_customer")
builder.add_edge("load_customer", "assess_risk")

builder.add_conditional_edges("assess_risk", choose_summary_path)

builder.add_edge("low_risk_summary", END)
builder.add_edge("high_risk_summary", END)

graph = builder.compile()


def run_example(customer_name: str):
    print(f"\n=== Running graph for: {customer_name} ===")

    final_state = graph.invoke(
        {
            "customer_name": customer_name,
            "credit_score": 0,
            "debt_to_income": 0.0,
            "risk_level": "unknown",
            "message": "",
        }
    )

    print("\nFinal state:")
    print(final_state)


if __name__ == "__main__":
    run_example("alice")
    run_example("ben")
