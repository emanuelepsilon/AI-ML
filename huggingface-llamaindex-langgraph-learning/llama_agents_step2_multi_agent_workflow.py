"""LlamaIndex multi-agent workflow with specialist handoffs."""

import asyncio

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0,
)

CUSTOMERS = {
    "C001": {
        "name": "Alice Rivera",
        "loan_type": "mortgage",
        "credit_score": 720,
        "annual_income": 52000,
        "monthly_debt": 1343,
        "debt_to_income": 0.31,
    },
    "C002": {
        "name": "Ben Carter",
        "loan_type": "personal loan",
        "credit_score": 610,
        "annual_income": 39000,
        "monthly_debt": 1527,
        "debt_to_income": 0.47,
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
}


def find_customer(customer_id_or_name: str):
    """Find a customer by ID, full name, or first name."""
    normalized_query = customer_id_or_name.lower().strip()

    for customer_id, record in CUSTOMERS.items():
        full_name = record["name"].lower()
        first_name = full_name.split()[0]

        if normalized_query in {customer_id.lower(), full_name, first_name}:
            return customer_id, record

    return None, None


def get_customer_profile(customer_id_or_name: str) -> dict:
    """Get one customer's profile by ID or name."""
    customer_id, record = find_customer(customer_id_or_name)

    if record is None:
        return {
            "found": False,
            "message": f"No customer found for {customer_id_or_name!r}.",
        }

    return {
        "found": True,
        "customer_id": customer_id,
        **record,
    }


def get_loan_policy(loan_type: str) -> dict:
    """Get loan policy limits by loan type."""
    normalized_loan_type = loan_type.lower().strip()
    policy = LOAN_POLICIES.get(normalized_loan_type)

    if policy is None:
        return {
            "found": False,
            "message": f"No policy found for {loan_type!r}.",
            "available_loan_types": list(LOAN_POLICIES),
        }

    return {
        "found": True,
        "loan_type": normalized_loan_type,
        **policy,
    }


bank_tools = [
    FunctionTool.from_defaults(
        fn=get_customer_profile,
        name="get_customer_profile",
        description="Get bank customer profile by customer ID, full name, or first name.",
    ),
    FunctionTool.from_defaults(
        fn=get_loan_policy,
        name="get_loan_policy",
        description="Get bank loan policy limits for mortgage or personal loan.",
    ),
]


def calculate_debt_to_income(monthly_debt: float, annual_income: float) -> float:
    """Calculate debt-to-income ratio from monthly debt and annual income."""
    monthly_income = annual_income / 12
    return round(monthly_debt / monthly_income, 4)


def subtract(left: float, right: float) -> float:
    """Subtract right from left."""
    return round(left - right, 4)


math_tools = [
    FunctionTool.from_defaults(
        fn=calculate_debt_to_income,
        name="calculate_debt_to_income",
        description="Calculate debt-to-income ratio using monthly debt and annual income.",
    ),
    FunctionTool.from_defaults(
        fn=subtract,
        name="subtract",
        description="Subtract one number from another.",
    ),
]

bank_agent = ReActAgent(
    name="bank_agent",
    description=(
        "Handles bank customer profiles and loan policy lookups. "
        "Can retrieve customer records and loan policy limits."
    ),
    tools=bank_tools,
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are a bank data specialist. "
        "Use bank tools for customer and policy facts. "
        "Never make a final approval or rejection decision. "
        "If calculation is needed, say that the math_agent should handle it."
    ),
)

math_agent = ReActAgent(
    name="math_agent",
    description=(
        "Handles numeric calculations such as debt-to-income ratios and differences."
    ),
    tools=math_tools,
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are a math specialist. "
        "Use math tools for calculations. "
        "Do not invent bank facts. "
        "If bank data is needed, say that the bank_agent should handle it."
    ),
)

bank_workflow = AgentWorkflow(
    agents=[bank_agent, math_agent],
    root_agent="bank_agent",
    timeout=180,
)

math_workflow = AgentWorkflow(
    agents=[bank_agent, math_agent],
    root_agent="math_agent",
    timeout=180,
)


async def main():
    questions = [
        (
            "Bank question",
            "What is Alice's customer profile and credit score?",
            bank_workflow,
        ),
        (
            "Math question",
            (
                "Calculate the debt-to-income ratio for monthly debt 1343 "
                "and annual income 52000."
            ),
            math_workflow,
        ),
    ]

    try:
        for label, question, active_workflow in questions:
            print(f"\n--- {label} ---")
            print(f"Question: {question}")

            response = await active_workflow.run(user_msg=question)

            print("\nFinal response:")
            print(response)
    except ConnectionError:
        print("Could not connect to Ollama.")
        print("Start Ollama in another terminal, then run this file again:")
        print("    ollama serve")
        return


if __name__ == "__main__":
    asyncio.run(main())
