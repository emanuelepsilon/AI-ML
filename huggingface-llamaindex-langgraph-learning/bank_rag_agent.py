from difflib import SequenceMatcher

from smolagents import LiteLLMModel
from smolagents.models import ChatMessage, MessageRole


BANK_SYSTEM_PROMPT = """
You are a polite banking database assistant running locally with Ollama/Qwen.
Welcome the user warmly to the bank database when appropriate.
Act like a normal helpful LLM in conversation.

For bank/customer/loan/risk questions, answer only from the retrieved database records.
Do not invent customer facts, loan rules, credit scores, or policy details.
If the retrieved records are ambiguous, explain what matched and ask for clarification.
Never make a final credit approval or rejection decision.
Explain risk factors clearly and reference record IDs when useful.
If the retrieved context includes a retrieval note, respect it.
""".strip()


BANK_RECORDS = [
    {
        "type": "loan_product",
        "id": "HOME-30Y-FIXED",
        "name": "30-Year Fixed Mortgage",
        "aliases": ["mortgage", "home loan", "fixed mortgage", "30 year"],
        "text": (
            "30-year fixed mortgage. Minimum credit score 680. Maximum "
            "debt-to-income ratio 43%. Requires verified income and property appraisal."
        ),
    },
    {
        "type": "loan_product",
        "id": "SMB-WORKING-CAPITAL",
        "name": "Small Business Working Capital Loan",
        "aliases": ["small business", "business loan", "working capital", "smb"],
        "text": (
            "Small business working capital loan. Requires 2 years business history, "
            "positive cash flow, and no bankruptcy in the last 5 years."
        ),
    },
    {
        "type": "customer_profile",
        "id": "CUST-1001",
        "name": "Alice Rivera",
        "aliases": ["alice", "alice rivera", "cust-1001"],
        "text": (
            "Customer Alice Rivera. Credit score 720. Debt-to-income ratio 38%. "
            "Stable W2 income for 4 years. Interested in a mortgage."
        ),
    },
    {
        "type": "customer_profile",
        "id": "CUST-1002",
        "name": "Ben Carter",
        "aliases": ["ben", "ben carter", "cust-1002"],
        "text": (
            "Customer Ben Carter. Credit score 610. Debt-to-income ratio 51%. "
            "Recent missed payments. Interested in a personal loan."
        ),
    },
    {
        "type": "risk_policy",
        "id": "RISK-DTI",
        "name": "Debt-to-Income Risk Policy",
        "aliases": ["dti", "debt to income", "debt-to-income", "manual underwriting"],
        "text": (
            "Applicants with debt-to-income ratio above 45% should be flagged "
            "for manual underwriting review."
        ),
    },
    {
        "type": "risk_policy",
        "id": "RISK-CREDIT",
        "name": "Credit Score Risk Policy",
        "aliases": ["credit", "credit score", "high risk", "collateral"],
        "text": (
            "Applicants below credit score 640 are high risk unless secured "
            "by strong collateral or compensating factors."
        ),
    },
]


STOP_WORDS = {
    "a",
    "about",
    "ability",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "data",
    "database",
    "do",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "know",
    "like",
    "me",
    "of",
    "on",
    "or",
    "our",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "them",
    "there",
    "to",
    "we",
    "what",
    "who",
    "with",
    "you",
}

SYNONYMS = {
    "client": "customer",
    "clients": "customer",
    "customer": "customer",
    "customers": "customer",
    "people": "customer",
    "person": "customer",
    "borrower": "customer",
    "borrowers": "customer",
    "account": "customer",
    "accounts": "customer",
    "applicant": "customer",
    "applicants": "customer",
    "loan": "loan",
    "loans": "loan",
    "product": "loan",
    "products": "loan",
    "policy": "risk",
    "policies": "risk",
    "risk": "risk",
    "risks": "risk",
    "rule": "risk",
    "rules": "risk",
    "underwrite": "risk",
    "underwriting": "risk",
}

FOLLOW_UP_TERMS = {
    "he",
    "her",
    "him",
    "his",
    "it",
    "she",
    "that",
    "their",
    "them",
    "they",
    "this",
}

BANK_TERMS = {
    "alice",
    "applicant",
    "applicants",
    "bank",
    "ben",
    "borrower",
    "borrowers",
    "client",
    "clients",
    "customer",
    "customers",
    "database",
    "dti",
    "loan",
    "loans",
    "mortgage",
    "policy",
    "risk",
    "underwriting",
}


def normalize(text):
    return "".join(char.lower() if char.isalnum() else " " for char in text)


def tokenize(text):
    words = normalize(text).split()
    return {SYNONYMS.get(word, word) for word in words if word not in STOP_WORDS}


def words(text):
    return normalize(text).split()


def record_text(record):
    aliases = " ".join(record["aliases"])
    return f"{record['type']} {record['id']} {record['name']} {aliases} {record['text']}"


def format_records(records):
    if not records:
        return "No matching bank records found."

    return "\n\n".join(
        f"[{record['type']}] {record['id']} - {record['name']}\n{record['text']}"
        for record in records
    )


def chat_message(role, text):
    return ChatMessage(role=role, content=[{"type": "text", "text": text}])


def response_text(response):
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def is_bank_question(question):
    question_words = set(words(question))
    if question_words & BANK_TERMS:
        return True
    return bool(find_referenced_records(question))


def records_by_type(record_type):
    return [record for record in BANK_RECORDS if record["type"] == record_type]


def unique_records(records):
    return list({record["id"]: record for record in records}.values())


def fuzzy_contains(needle, haystack, threshold=0.8):
    if any(char.isdigit() for char in needle):
        return needle in haystack

    if needle in haystack:
        return True

    needle_words = needle.split()
    haystack_words = haystack.split()
    if len(needle_words) == 1:
        return any(
            SequenceMatcher(None, needle_words[0], word).ratio() >= threshold
            for word in haystack_words
        )

    window_size = len(needle_words)
    for index in range(len(haystack_words) - window_size + 1):
        window = " ".join(haystack_words[index : index + window_size])
        if SequenceMatcher(None, needle, window).ratio() >= threshold:
            return True
    return False


def find_referenced_records(question):
    normalized_question = normalize(question)
    matches = []

    for record in BANK_RECORDS:
        aliases = [record["id"], record["name"], *record["aliases"]]
        normalized_aliases = [normalize(alias).strip() for alias in aliases]
        if any(alias in normalized_question for alias in normalized_aliases):
            matches.append(record)
            continue

        fuzzy_aliases = [
            alias for alias in normalized_aliases if not any(char.isdigit() for char in alias)
        ]
        if any(fuzzy_contains(alias, normalized_question) for alias in fuzzy_aliases):
            matches.append(record)

    return unique_records(matches)


def infer_requested_types(question):
    question_tokens = tokenize(question)
    requested_types = set()

    if "customer" in question_tokens:
        requested_types.add("customer_profile")
    if "loan" in question_tokens or question_tokens & {"mortgage", "business", "smb"}:
        requested_types.add("loan_product")
    if "risk" in question_tokens or question_tokens & {"credit", "dti", "collateral"}:
        requested_types.add("risk_policy")

    return requested_types


def wants_overview(question):
    normalized_question = normalize(question)
    question_words = set(words(question))
    overview_phrases = [
        "what is included",
        "what information",
        "what do we have",
        "what records",
        "what data",
        "show database",
        "list database",
        "entire database",
        "all records",
    ]
    return "database" in question_words and (
        bool(question_words & {"all", "include", "included", "inside", "list", "overview", "show", "what"})
        or any(phrase in normalized_question for phrase in overview_phrases)
    )


def wants_customer_list(question):
    question_words = set(words(question))
    question_tokens = tokenize(question)
    customerish = bool(question_tokens & {"customer"} or question_words & {"who", "people"})
    listish = bool(question_words & {"all", "list", "name", "names", "who", "people", "clients", "customers"})
    return customerish and listish


def wants_comparison(question):
    question_words = set(words(question))
    return bool(question_words & {"compare", "comparison", "versus", "vs", "between"})


def wants_risk_review(question):
    question_tokens = tokenize(question)
    question_words = set(words(question))
    return bool(
        question_tokens & {"risk"}
        or question_words & {"review", "assess", "evaluate", "eligible", "eligibility", "qualify", "qualified"}
    )


def expand_context(question, records):
    expanded = list(records)
    requested_types = infer_requested_types(question)

    if wants_comparison(question) and not records:
        expanded.extend(records_by_type("customer_profile"))

    if wants_risk_review(question):
        expanded.extend(records_by_type("risk_policy"))

    if "loan_product" in requested_types and not any(
        record["type"] == "loan_product" for record in expanded
    ):
        expanded.extend(records_by_type("loan_product"))

    if "risk_policy" in requested_types and not any(
        record["type"] == "risk_policy" for record in expanded
    ):
        expanded.extend(records_by_type("risk_policy"))

    return unique_records(expanded)


def retrieve_bank_records(question, previous_records=None):
    normalized_question = normalize(question)
    question_tokens = tokenize(question)
    question_words = set(words(question))

    if question_words & FOLLOW_UP_TERMS and previous_records:
        records = expand_context(question, previous_records)
        return records, "follow-up question using previous retrieved records"

    referenced_records = find_referenced_records(question)

    if wants_overview(question):
        requested_types = infer_requested_types(question)
        if requested_types:
            records = [
                record for record in BANK_RECORDS if record["type"] in requested_types
            ]
            return records, "typed database overview requested"
        return BANK_RECORDS, "full database overview requested"

    if wants_customer_list(question):
        customers = [record for record in BANK_RECORDS if record["type"] == "customer_profile"]
        return customers, "customer list requested"

    if referenced_records:
        records = expand_context(question, referenced_records)
        return records, "specific record match with contextual expansion"

    requested_types = infer_requested_types(question)
    if requested_types and question_words & {"all", "available", "include", "included", "list", "show", "what"}:
        records = [record for record in BANK_RECORDS if record["type"] in requested_types]
        return records, "requested record category"

    scored_records = []
    for record in BANK_RECORDS:
        tokens = tokenize(record_text(record))
        score = len(question_tokens & tokens)
        if score:
            scored_records.append((score, record))

    scored_records.sort(key=lambda item: item[0], reverse=True)
    records = [record for _, record in scored_records[:5]]
    records = expand_context(question, records)

    if not records and is_bank_question(question):
        return BANK_RECORDS, "bank question with no precise match; full database provided for clarification"

    return records, "keyword and synonym match"


def model_answer(model, question, records, retrieval_reason):
    context = format_records(records)
    messages = [
        chat_message(MessageRole.SYSTEM, BANK_SYSTEM_PROMPT),
        chat_message(
            MessageRole.USER,
            (
                f"User question: {question}\n\n"
                f"Retrieval reason: {retrieval_reason}\n\n"
                f"Retrieved bank records:\n{context}\n\n"
                "Answer naturally and concisely. "
                "If several records are retrieved, separate what is known from what needs clarification. "
                "If records are missing, say so."
            ),
        ),
    ]
    response = model.generate(messages)
    return response_text(response)


def normal_answer(model, question):
    messages = [
        chat_message(MessageRole.SYSTEM, BANK_SYSTEM_PROMPT),
        chat_message(MessageRole.USER, question),
    ]
    response = model.generate(messages)
    return response_text(response)


def main():
    model = LiteLLMModel(
        model_id="ollama/qwen2:7b",
        api_base="http://localhost:11434",
        num_ctx=8192,
    )

    print("Welcome to the Bank Database Assistant.")
    print("I can help with customers, loans, risk policies, and underwriting notes.")
    print("Type 'exit' to stop.\n")

    previous_records = []

    while True:
        question = input("bank> ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Stopping Bank Database Assistant.")
            break

        if not question:
            continue

        if is_bank_question(question):
            records, reason = retrieve_bank_records(question, previous_records)
            previous_records = records
            answer = model_answer(model, question, records, reason)
        else:
            answer = normal_answer(model, question)

        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
