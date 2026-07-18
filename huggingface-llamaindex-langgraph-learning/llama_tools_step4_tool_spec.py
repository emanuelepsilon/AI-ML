"""LlamaIndex tool specification for grouped banking tools."""

import asyncio
import json

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.llms.ollama import Ollama

CUSTOMERS = {
    "C001": {
        "name": "Alice Rivera",
        "loan_type": "mortgage",
        "credit_score": 720,
        "income": 52000,
        "debt_to_income": 0.31,
        "notes": "Stable employment and clean payment history.",
    },
    "C002": {
        "name": "Ben Carter",
        "loan_type": "personal loan",
        "credit_score": 610,
        "income": 39000,
        "debt_to_income": 0.47,
        "notes": "Several late payments in the last 18 months.",
    },
    "C003": {
        "name": "Maya Lind",
        "loan_type": "small business loan",
        "credit_score": 680,
        "income": 76000,
        "debt_to_income": 0.38,
        "notes": "Business revenue is growing, but cashflow is seasonal.",
    },
}

LOAN_POLICIES = {
    "mortgage": {
        "minimum_credit_score": 680,
        "maximum_debt_to_income": 0.36,
        "human_review_required": True,
    },
    "personal loan": {
        "minimum_credit_score": 640,
        "maximum_debt_to_income": 0.40,
        "human_review_required": True,
    },
    "small business loan": {
        "minimum_credit_score": 660,
        "maximum_debt_to_income": 0.45,
        "human_review_required": True,
    },
}


def find_customer(customer_id_or_name: str) -> tuple[str | None, dict | None]:
    """
    Find a customer by exact customer ID or by name.

    This makes the tools more flexible because users usually say "Alice",
    while databases usually store "C001".
    """
    normalized_query = customer_id_or_name.lower().strip()

    for customer_id, record in CUSTOMERS.items():
        name = record["name"].lower()
        first_name = name.split()[0]

        if normalized_query in {customer_id.lower(), name, first_name}:
            return customer_id, record

    return None, None


class BankToolSpec(BaseToolSpec):
    spec_functions = [
        "list_customers",
        "get_customer_profile",
        "get_loan_policy",
        "compare_customer_to_loan_policy",
    ]

    def list_customers(self) -> str:
        """
        List all customers in the bank database with their customer IDs.
        Use this first when the user asks who is in the database.
        """
        rows = [
            {
                "customer_id": customer_id,
                "name": record["name"],
                "loan_type": record["loan_type"],
            }
            for customer_id, record in CUSTOMERS.items()
        ]
        return json.dumps(rows, indent=2)

    def get_customer_profile(self, customer_id: str) -> str:
        """
        Get one customer's full profile by customer ID or customer name.
        Use this when the user asks about a specific customer.
        """
        matched_customer_id, record = find_customer(customer_id)

        if record is None:
            return (
                f"No customer found for customer_id_or_name={customer_id!r}. "
                "Call list_customers if you need valid customer IDs."
            )

        response = {"customer_id": matched_customer_id, **record}
        return json.dumps(response, indent=2)

    def get_loan_policy(self, loan_type: str) -> str:
        """
        Get the bank policy for a loan type.
        Use this when the user asks about rules, risk limits, or policy.
        """
        normalized_loan_type = loan_type.lower().strip()
        policy = LOAN_POLICIES.get(normalized_loan_type)

        if policy is None:
            return (
                f"No policy found for loan_type={loan_type!r}. "
                f"Available loan types: {', '.join(LOAN_POLICIES)}."
            )

        response = {"loan_type": normalized_loan_type, **policy}
        return json.dumps(response, indent=2)

    def compare_customer_to_loan_policy(self, customer_id: str, loan_type: str) -> str:
        """
        Compare one customer's profile to one loan policy using exact rules.
        Use this when the user asks for risk factors or policy comparison.
        """
        normalized_loan_type = loan_type.lower().strip()

        matched_customer_id, customer = find_customer(customer_id)
        policy = LOAN_POLICIES.get(normalized_loan_type)

        if customer is None:
            return (
                f"No customer found for customer_id_or_name={customer_id!r}. "
                "Call list_customers if you need valid customer IDs."
            )

        if policy is None:
            return (
                f"No policy found for loan_type={loan_type!r}. "
                f"Available loan types: {', '.join(LOAN_POLICIES)}."
            )

        credit_score_meets_policy = (
            customer["credit_score"] >= policy["minimum_credit_score"]
        )
        debt_to_income_meets_policy = (
            customer["debt_to_income"] <= policy["maximum_debt_to_income"]
        )

        comparison = {
            "customer_id": matched_customer_id,
            "customer_name": customer["name"],
            "loan_type": normalized_loan_type,
            "credit_score": customer["credit_score"],
            "minimum_credit_score": policy["minimum_credit_score"],
            "credit_score_meets_policy": credit_score_meets_policy,
            "debt_to_income": customer["debt_to_income"],
            "maximum_debt_to_income": policy["maximum_debt_to_income"],
            "debt_to_income_meets_policy": debt_to_income_meets_policy,
            "human_review_required": policy["human_review_required"],
            "important_note": (
                "This is only a policy comparison, not a final approval or "
                "rejection decision."
            ),
        }
        return json.dumps(comparison, indent=2)


bank_tool_spec = BankToolSpec()
bank_tools = bank_tool_spec.to_tool_list()

llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

agent = ReActAgent(
    tools=bank_tools,
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are a polite bank database assistant. "
        "Use the provided tools for bank facts. "
        "Do not invent customer records or loan policies. "
        "Never make final approval or rejection decisions. "
        "If information is missing, say what is missing."
    ),
)


async def main():
    question = (
        "For Alice, show her customer record and compare her mortgage details "
        "to the mortgage policy using exact policy checks. "
        "Do not approve or reject the loan."
    )

    response = await agent.run(question)

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
