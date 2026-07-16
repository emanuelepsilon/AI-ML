"""
LangGraph - Step 1: Building blocks

This follows the Hugging Face page:

    1. State
    2. Nodes
    3. Edges
    4. StateGraph

Main idea:
LangGraph is a graph-based workflow system.

Compared to LlamaIndex Workflow:

    LlamaIndex Workflow:
        Event -> step -> Event -> step

    LangGraph:
        State -> node -> updated State -> next node

The graph decides where to go next using edges.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
# State is the shared data that flows through the graph.
#
# Every node receives the current state.
# Every node returns updates to the state.
#
# Think of this as the graph's shared notebook.
class BankState(TypedDict):
    customer_name: str
    credit_score: int
    debt_to_income: float
    risk_level: str
    message: str


# ---------------------------------------------------------------------------
# 2. NODES
# ---------------------------------------------------------------------------
# Nodes are just Python functions.
#
# Each node:
# - receives state
# - does some work
# - returns a dictionary of state updates
#
# It does NOT need to return the whole state.
# It can return only the fields it wants to update.
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
            state["message"]
            + " Route: high-risk summary. Escalate to human review."
        )
    }


# ---------------------------------------------------------------------------
# 3. CONDITIONAL EDGE FUNCTION
# ---------------------------------------------------------------------------
# Edges decide where the graph goes next.
#
# A conditional edge function reads the current state and returns the name of
# the next node.
#
# Here:
# - low risk goes to low_risk_summary
# - high risk goes to high_risk_summary
def choose_summary_path(state: BankState) -> Literal[
    "low_risk_summary",
    "high_risk_summary",
]:
    print("--- choose_summary_path edge function ---")

    if state["risk_level"] == "high":
        return "high_risk_summary"

    return "low_risk_summary"


# ---------------------------------------------------------------------------
# 4. BUILD STATEGRAPH
# ---------------------------------------------------------------------------
# StateGraph is the container.
#
# We add:
# - nodes
# - normal edges
# - conditional edges
builder = StateGraph(BankState)

builder.add_node("load_customer", load_customer)
builder.add_node("assess_risk", assess_risk)
builder.add_node("low_risk_summary", low_risk_summary)
builder.add_node("high_risk_summary", high_risk_summary)

# Direct edges:
# START always goes to load_customer.
# load_customer always goes to assess_risk.
builder.add_edge(START, "load_customer")
builder.add_edge("load_customer", "assess_risk")

# Conditional edge:
# assess_risk decides which summary node comes next.
builder.add_conditional_edges("assess_risk", choose_summary_path)

# Both summary nodes end the graph.
builder.add_edge("low_risk_summary", END)
builder.add_edge("high_risk_summary", END)

# Compile turns the builder into a runnable graph.
graph = builder.compile()


# ---------------------------------------------------------------------------
# 5. RUN THE GRAPH
# ---------------------------------------------------------------------------
# We invoke the same graph twice:
# - Alice should follow the low-risk route.
# - Ben should follow the high-risk route.
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
