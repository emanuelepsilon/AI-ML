import csv
import io
import json
import re
from datetime import date

from sqlalchemy.orm import Session

from app.models import Company, Invoice, Transaction
from app.services.categorization import categorizer


FIELD_PATTERNS = {
    "invoice_number": r"(?:Invoice|Invoice Number|Reference)\s*[:#]\s*([^\n]+)",
    "company_external_id": r"(?:Company|Customer ID|Bill To)\s*:\s*([^\n]+)",
    "vendor_name": r"(?:Vendor|Supplier|From)\s*:\s*([^\n]+)",
    "issue_date": r"(?:Issue Date|Invoice Date|Issued)\s*:\s*(\d{4}-\d{2}-\d{2})",
    "due_date": r"(?:Due Date|Payment Due|Due)\s*:\s*(\d{4}-\d{2}-\d{2})",
    "amount": r"(?:Total|Amount Due|Invoice Total)\s*:\s*(?:[A-Z]{3}\s*)?([\d,.]+)",
    "currency": r"(?:Currency|Total|Amount Due|Invoice Total)\s*:\s*([A-Z]{3})",
    "description": r"(?:Description|Service|Details)\s*:\s*([^\n]+)",
}


def parse_invoice_document(text):
    parsed = {}
    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed[field] = match.group(1).strip()
    if "amount" in parsed:
        parsed["amount"] = float(parsed["amount"].replace(",", ""))
    if "currency" not in parsed:
        currency = re.search(r"\b(SEK|EUR|USD|GBP)\b", text)
        parsed["currency"] = currency.group(1) if currency else "SEK"
    return parsed


def _csv_rows(content):
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def import_companies(session: Session, content):
    imported = 0
    for row in _csv_rows(content):
        if session.query(Company).filter_by(external_id=row["company_external_id"]).first():
            continue
        session.add(
            Company(
                external_id=row["company_external_id"],
                name=row["name"],
                industry=row["industry"],
                country=row["country"],
                currency=row["currency"],
            )
        )
        imported += 1
    session.commit()
    return imported


def import_invoice_documents(session: Session, content):
    imported = 0
    for line in content.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        parsed = parse_invoice_document(record["text"])
        number = parsed.get("invoice_number")
        if not number or session.query(Invoice).filter_by(invoice_number=number).first():
            continue
        company = session.query(Company).filter_by(
            external_id=parsed["company_external_id"]
        ).one()
        expected = record.get("expected", {})
        description = parsed.get("description", "")
        session.add(
            Invoice(
                invoice_number=number,
                company_id=company.id,
                vendor_name=parsed["vendor_name"],
                issue_date=date.fromisoformat(parsed["issue_date"]),
                due_date=date.fromisoformat(parsed["due_date"]),
                amount=float(parsed["amount"]),
                currency=parsed["currency"],
                description=description,
                predicted_category=categorizer.predict(
                    f"{parsed['vendor_name']} {description}"
                ),
                expected_category=expected.get("category"),
                raw_document=record["text"],
            )
        )
        imported += 1
    session.commit()
    return imported


def import_transactions(session: Session, content):
    imported = 0
    for row in _csv_rows(content):
        if session.query(Transaction).filter_by(external_id=row["transaction_external_id"]).first():
            continue
        company = session.query(Company).filter_by(
            external_id=row["company_external_id"]
        ).one()
        expected = row.get("expected_anomaly", "").strip().lower()
        session.add(
            Transaction(
                external_id=row["transaction_external_id"],
                company_id=company.id,
                booking_date=date.fromisoformat(row["booking_date"]),
                amount=float(row["amount"]),
                currency=row["currency"],
                counterparty=row["counterparty"],
                reference=row["reference"],
                expected_anomaly=expected == "true" if expected else None,
                ground_truth_invoice_number=row.get("ground_truth_invoice_number") or None,
            )
        )
        imported += 1
    session.commit()
    return imported

