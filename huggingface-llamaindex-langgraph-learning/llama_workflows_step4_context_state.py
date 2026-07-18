"""LlamaIndex workflow using Context for shared execution state."""

import asyncio

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

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

POLICIES = {
    "mortgage": {
        "minimum_credit_score": 680,
        "maximum_debt_to_income": 0.36,
    },
    "personal loan": {
        "minimum_credit_score": 640,
        "maximum_debt_to_income": 0.40,
    },
}


class CustomerLoadedEvent(Event):
    pass


class CreditCheckedEvent(Event):
    pass


class DebtCheckedEvent(Event):
    pass


class ContextStateRiskWorkflow(Workflow):
    @step
    async def load_customer(
        self, ctx: Context, ev: StartEvent
    ) -> CustomerLoadedEvent | StopEvent:
        customer_name = ev.get("customer_name", "").lower().strip()
        customer = CUSTOMERS.get(customer_name)

        if customer is None:
            return StopEvent(result=f"No customer found for {customer_name!r}.")

        policy = POLICIES[customer["loan_type"]]

        await ctx.store.set("customer", customer)
        await ctx.store.set("policy", policy)
        await ctx.store.set("risk_flags", [])
        await ctx.store.set("checks_performed", 0)

        print(f"Loaded customer into Context: {customer['name']}")
        return CustomerLoadedEvent()

    @step
    async def check_credit(
        self, ctx: Context, ev: CustomerLoadedEvent
    ) -> CreditCheckedEvent:
        customer = await ctx.store.get("customer")
        policy = await ctx.store.get("policy")
        risk_flags = await ctx.store.get("risk_flags")
        checks_performed = await ctx.store.get("checks_performed")

        if customer["credit_score"] < policy["minimum_credit_score"]:
            risk_flags.append(
                f"credit score {customer['credit_score']} is below "
                f"minimum {policy['minimum_credit_score']}"
            )

        checks_performed += 1

        await ctx.store.set("risk_flags", risk_flags)
        await ctx.store.set("checks_performed", checks_performed)

        print("Credit check completed.")
        return CreditCheckedEvent()

    @step
    async def check_debt_to_income(
        self, ctx: Context, ev: CreditCheckedEvent
    ) -> DebtCheckedEvent:
        customer = await ctx.store.get("customer")
        policy = await ctx.store.get("policy")
        risk_flags = await ctx.store.get("risk_flags")
        checks_performed = await ctx.store.get("checks_performed")

        if customer["debt_to_income"] > policy["maximum_debt_to_income"]:
            risk_flags.append(
                f"debt-to-income {customer['debt_to_income']} is above "
                f"maximum {policy['maximum_debt_to_income']}"
            )

        checks_performed += 1

        await ctx.store.set("risk_flags", risk_flags)
        await ctx.store.set("checks_performed", checks_performed)

        print("Debt-to-income check completed.")
        return DebtCheckedEvent()

    @step
    async def summarize(self, ctx: Context, ev: DebtCheckedEvent) -> StopEvent:
        customer = await ctx.store.get("customer")
        risk_flags = await ctx.store.get("risk_flags")
        checks_performed = await ctx.store.get("checks_performed")

        if not risk_flags:
            risk_summary = "No policy risk flags found."
        else:
            risk_summary = risk_flags

        return StopEvent(
            result={
                "customer_id": customer["customer_id"],
                "name": customer["name"],
                "loan_type": customer["loan_type"],
                "checks_performed": checks_performed,
                "risk_summary": risk_summary,
                "final_decision": "Not provided. Human review required.",
            }
        )


async def main():
    workflow = ContextStateRiskWorkflow(timeout=10, verbose=True)

    for customer_name in ["alice", "ben"]:
        print(f"\n--- Running workflow for: {customer_name} ---")
        result = await workflow.run(customer_name=customer_name)

        print("\nWorkflow result:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
