"""MCP server exposing local banking data tools."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="bank_mcp_server",
    instructions=(
        "Bank MCP server exposing customer and loan policy tools. "
        "Do not use these tools to make final credit decisions."
    ),
)

CUSTOMERS = {
    "C001": {
        "name": "Alice Rivera",
        "loan_type": "mortgage",
        "credit_score": 720,
        "income": 52000,
        "debt_to_income": 0.31,
    },
    "C002": {
        "name": "Ben Carter",
        "loan_type": "personal loan",
        "credit_score": 610,
        "income": 39000,
        "debt_to_income": 0.47,
    },
    "C003": {
        "name": "Maya Lind",
        "loan_type": "small business loan",
        "credit_score": 680,
        "income": 76000,
        "debt_to_income": 0.38,
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


def find_customer(customer_id_or_name: str):
    """Find a customer by ID, full name, or first name."""
    normalized_query = customer_id_or_name.lower().strip()

    for customer_id, record in CUSTOMERS.items():
        full_name = record["name"].lower()
        first_name = full_name.split()[0]

        if normalized_query in {customer_id.lower(), full_name, first_name}:
            return customer_id, record

    return None, None


@mcp.tool()
def list_customers() -> list[dict]:
    """List all customers in the bank database."""
    return [
        {
            "customer_id": customer_id,
            "name": record["name"],
            "loan_type": record["loan_type"],
        }
        for customer_id, record in CUSTOMERS.items()
    ]


@mcp.tool()
def get_customer_profile(customer_id_or_name: str) -> dict:
    """Get one customer's full profile by customer ID or name."""
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


@mcp.tool()
def get_loan_policy(loan_type: str) -> dict:
    """Get policy limits for a loan type."""
    normalized_loan_type = loan_type.lower().strip()
    policy = LOAN_POLICIES.get(normalized_loan_type)

    if policy is None:
        return {
            "found": False,
            "message": f"No loan policy found for {loan_type!r}.",
            "available_loan_types": list(LOAN_POLICIES),
        }

    return {
        "found": True,
        "loan_type": normalized_loan_type,
        **policy,
    }


@mcp.tool()
def compare_customer_to_policy(customer_id_or_name: str, loan_type: str) -> dict:
    """
    Compare a customer profile to policy limits.

    This is not a final approval or rejection.
    It only returns policy checks and risk flags.
    """
    customer_id, customer = find_customer(customer_id_or_name)
    normalized_loan_type = loan_type.lower().strip()
    policy = LOAN_POLICIES.get(normalized_loan_type)

    if customer is None:
        return {
            "found": False,
            "message": f"No customer found for {customer_id_or_name!r}.",
        }

    if policy is None:
        return {
            "found": False,
            "message": f"No policy found for {loan_type!r}.",
            "available_loan_types": list(LOAN_POLICIES),
        }

    return {
        "found": True,
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "loan_type": normalized_loan_type,
        "credit_score": customer["credit_score"],
        "minimum_credit_score": policy["minimum_credit_score"],
        "credit_score_meets_policy": (
            customer["credit_score"] >= policy["minimum_credit_score"]
        ),
        "debt_to_income": customer["debt_to_income"],
        "maximum_debt_to_income": policy["maximum_debt_to_income"],
        "debt_to_income_meets_policy": (
            customer["debt_to_income"] <= policy["maximum_debt_to_income"]
        ),
        "human_review_required": policy["human_review_required"],
        "final_decision": "Not provided. Human review required.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
