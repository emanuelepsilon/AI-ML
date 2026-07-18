"""LlamaIndex workflow using custom events."""

import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

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


class CustomerLoadedEvent(Event):
    customer: dict


class CustomerRiskWorkflow(Workflow):
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


async def main():
    workflow = CustomerRiskWorkflow(timeout=10, verbose=True)

    result = await workflow.run(customer_name="alice")

    print("\nWorkflow result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
