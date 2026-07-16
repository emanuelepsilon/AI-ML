"""
LlamaIndex Workflows - Step 2: Multi-step workflow with custom events

Idea:
The previous workflow was:

    StartEvent -> one step -> StopEvent

This workflow is:

    StartEvent -> load_customer -> CustomerLoadedEvent -> assess_risk -> StopEvent

Important:
A step does not have to end the workflow.
A step can return a custom Event that carries structured data to the next step.

That makes workflows feel like typed pipelines.
"""

import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


# ---------------------------------------------------------------------------
# 1. FAKE BANK DATABASE
# ---------------------------------------------------------------------------
# No LLM yet.
# No RAG yet.
#
# We are focusing only on workflow mechanics.
CUSTOMERS = {
    "alice": {
        "customer_id": "C001",
        "name": "Alice Rivera",
        "loan_type": "mortgage",
        "credit_score": 720,
        "debt_to_income": 0.31,
    },
    "ben": {
        "customer_id": "C002",
        "name": "Ben Carter",
        "loan_type": "personal loan",
        "credit_score": 610,
        "debt_to_income": 0.47,
    },
}


# ---------------------------------------------------------------------------
# 2. CUSTOM EVENT
# ---------------------------------------------------------------------------
# Event is the base class for workflow events.
#
# This custom event carries customer data from one step to another.
#
# In plain English:
# "The customer was successfully loaded, here is the customer data."
class CustomerLoadedEvent(Event):
    customer: dict


# ---------------------------------------------------------------------------
# 3. CREATE WORKFLOW CLASS
# ---------------------------------------------------------------------------
class CustomerRiskWorkflow(Workflow):
    # -----------------------------------------------------------------------
    # STEP 1: LOAD CUSTOMER
    # -----------------------------------------------------------------------
    # This step listens for StartEvent.
    #
    # StartEvent can carry arbitrary keyword arguments from workflow.run(...).
    # We will pass customer_name="alice" when running the workflow.
    #
    # This step returns CustomerLoadedEvent, not StopEvent.
    # That means the workflow continues.
    @step
    async def load_customer(self, ev: StartEvent) -> CustomerLoadedEvent | StopEvent:
        customer_name = ev.get("customer_name", "").lower().strip()
        customer = CUSTOMERS.get(customer_name)

        if customer is None:
            return StopEvent(
                result=f"No customer found for customer_name={customer_name!r}."
            )

        print(f"Loaded customer: {customer['name']}")

        return CustomerLoadedEvent(customer=customer)

    # -----------------------------------------------------------------------
    # STEP 2: ASSESS RISK
    # -----------------------------------------------------------------------
    # This step listens for CustomerLoadedEvent.
    #
    # It receives the structured data returned by load_customer(...).
    #
    # This step returns StopEvent, so the workflow ends here.
    @step
    async def assess_risk(self, ev: CustomerLoadedEvent) -> StopEvent:
        customer = ev.customer

        risk_flags = []

        if customer["credit_score"] < 640:
            risk_flags.append("credit score below 640")

        if customer["debt_to_income"] > 0.40:
            risk_flags.append("debt-to-income ratio above 0.40")

        if not risk_flags:
            risk_summary = "No simple risk flags found."
        else:
            risk_summary = "Risk flags: " + ", ".join(risk_flags)

        result = {
            "customer_id": customer["customer_id"],
            "name": customer["name"],
            "loan_type": customer["loan_type"],
            "credit_score": customer["credit_score"],
            "debt_to_income": customer["debt_to_income"],
            "risk_summary": risk_summary,
            "final_decision": "Not provided. Human review required.",
        }

        return StopEvent(result=result)


# ---------------------------------------------------------------------------
# 4. RUN THE WORKFLOW
# ---------------------------------------------------------------------------
async def main():
    workflow = CustomerRiskWorkflow(timeout=10, verbose=True)

    # customer_name becomes available inside StartEvent.
    result = await workflow.run(customer_name="alice")

    print("\nWorkflow result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
