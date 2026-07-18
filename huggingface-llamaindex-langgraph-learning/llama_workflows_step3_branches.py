"""LlamaIndex workflow with conditional branches."""

import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

CUSTOMERS = {
    "alice": {
        "customer_id": "C001",
        "name": "Alice Rivera",
        "credit_score": 720,
        "debt_to_income": 0.31,
    },
    "ben": {
        "customer_id": "C002",
        "name": "Ben Carter",
        "credit_score": 610,
        "debt_to_income": 0.47,
    },
}


class CustomerLoadedEvent(Event):
    customer: dict


class CustomerMissingEvent(Event):
    customer_name: str
    available_customers: list[str]


class BranchingRiskWorkflow(Workflow):
    @step
    async def load_customer(
        self, ev: StartEvent
    ) -> CustomerLoadedEvent | CustomerMissingEvent:
        customer_name = ev.get("customer_name", "").lower().strip()
        customer = CUSTOMERS.get(customer_name)

        if customer is None:
            print(f"Customer not found: {customer_name}")
            return CustomerMissingEvent(
                customer_name=customer_name,
                available_customers=list(CUSTOMERS.keys()),
            )

        print(f"Customer found: {customer['name']}")
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

        return StopEvent(
            result={
                "path": "customer_found",
                "customer_id": customer["customer_id"],
                "name": customer["name"],
                "risk_summary": risk_summary,
                "final_decision": "Not provided. Human review required.",
            }
        )

    @step
    async def handle_missing_customer(self, ev: CustomerMissingEvent) -> StopEvent:
        return StopEvent(
            result={
                "path": "customer_missing",
                "message": f"No customer found for {ev.customer_name!r}.",
                "available_customers": ev.available_customers,
            }
        )


async def main():
    workflow = BranchingRiskWorkflow(timeout=10, verbose=True)

    for customer_name in ["alice", "zoe"]:
        print(f"\n--- Running workflow for: {customer_name} ---")
        result = await workflow.run(customer_name=customer_name)

        print("\nWorkflow result:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
