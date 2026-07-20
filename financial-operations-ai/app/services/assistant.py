import re

import httpx
from sqlalchemy.orm import Session

from app.config import llm_config
from app.models import Anomaly, Company, Invoice, Reconciliation, Transaction


def _source(source_id, kind, title, content):
    return {"id": source_id, "type": kind, "title": title, "content": content}


def retrieve_evidence(session: Session, question: str, limit=6):
    lowered = question.lower()
    sources = []
    invoice_numbers = re.findall(r"\bINV-\d{4}-\d{3,4}\b", question.upper())
    for number in invoice_numbers:
        invoice = session.query(Invoice).filter_by(invoice_number=number).first()
        if invoice:
            sources.append(
                _source(
                    f"invoice:{invoice.id}",
                    "invoice",
                    invoice.invoice_number,
                    (
                        f"Vendor {invoice.vendor_name}; amount {invoice.amount:.2f} {invoice.currency}; "
                        f"due {invoice.due_date}; category {invoice.predicted_category}."
                    ),
                )
            )

    if any(term in lowered for term in ["anomal", "unusual", "risk", "flag"]):
        rows = (
            session.query(Anomaly, Transaction, Company)
            .join(Transaction, Anomaly.transaction_id == Transaction.id)
            .join(Company, Transaction.company_id == Company.id)
            .order_by(Anomaly.score.desc())
            .limit(limit)
            .all()
        )
        for anomaly, transaction, company in rows:
            sources.append(
                _source(
                    f"anomaly:{anomaly.id}",
                    "anomaly",
                    f"{transaction.external_id} at {company.name}",
                    (
                        f"Score {anomaly.score:.2f}; severity {anomaly.severity}; "
                        f"amount {abs(transaction.amount):.2f} {transaction.currency}; "
                        f"counterparty {transaction.counterparty}; reasons {anomaly.reasons}."
                    ),
                )
            )

    if any(term in lowered for term in ["match", "reconcil", "payment", "invoice"]):
        rows = (
            session.query(Reconciliation, Invoice, Transaction)
            .join(Invoice, Reconciliation.invoice_id == Invoice.id)
            .join(Transaction, Reconciliation.transaction_id == Transaction.id)
            .order_by(Reconciliation.score.desc())
            .limit(limit)
            .all()
        )
        for match, invoice, transaction in rows:
            sources.append(
                _source(
                    f"reconciliation:{match.id}",
                    "reconciliation",
                    f"{invoice.invoice_number} to {transaction.external_id}",
                    (
                        f"Score {match.score:.2f}; status {match.status}; invoice amount "
                        f"{invoice.amount:.2f} {invoice.currency}; {match.rationale}"
                    ),
                )
            )

    unique = {item["id"]: item for item in sources}
    return list(unique.values())[:limit]


def _offline_answer(question, sources):
    if not sources:
        return (
            "The available records do not contain enough evidence to answer that question. "
            "Ask about an invoice number, reconciliation or anomaly."
        )
    lines = ["The records support the following findings:"]
    for item in sources[:4]:
        lines.append(f"• {item['content']} [{item['id']}]")
    lines.append("Any financial action still requires human review.")
    return "\n".join(lines)


def _llm_answer(question, sources, config):
    evidence = "\n".join(
        f"[{item['id']}] {item['title']}: {item['content']}" for item in sources
    )
    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You investigate financial operations. Answer only from the supplied evidence. "
                    "Cite every factual statement with the exact source ID in square brackets. "
                    "Do not approve transactions or invent missing facts."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{evidence}"},
        ],
        "temperature": 0,
    }
    response = httpx.post(
        f"{config['base_url'].rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config['api_key']}"},
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def answer_question(session: Session, question: str):
    sources = retrieve_evidence(session, question)
    config = llm_config()
    if sources and config["api_key"] and config["model"]:
        try:
            return _llm_answer(question, sources, config), sources, "llm"
        except (httpx.HTTPError, KeyError, IndexError):
            pass
    return _offline_answer(question, sources), sources, "deterministic"

