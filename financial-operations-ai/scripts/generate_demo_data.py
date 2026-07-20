import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np


COMPANIES = [
    ("COMP-001", "Nordic Fabrication AB", "Manufacturing"),
    ("COMP-002", "Lumen Analytics AB", "Technology"),
    ("COMP-003", "Baltic Field Services AB", "Engineering"),
    ("COMP-004", "Sundby Retail Group AB", "Retail"),
    ("COMP-005", "Northline Logistics AB", "Transport"),
    ("COMP-006", "Aster Consulting AB", "Consulting"),
]

CATEGORY_DATA = {
    "Software": [
        ("CloudGrid Nordic", "cloud hosting subscription"),
        ("SecureStack AB", "cybersecurity monitoring service"),
        ("DataCore Systems", "database hosting renewal"),
    ],
    "Travel": [
        ("Nordic Rail Travel", "train tickets for conference"),
        ("CityStay Hotels", "hotel accommodation for client visit"),
        ("Scandinavian Mobility", "rental car for site inspection"),
    ],
    "Office": [
        ("Workplace Supply AB", "office chairs and desks"),
        ("Paperhouse Nordic", "printer supplies and paper"),
        ("Device Depot", "computer peripherals for office"),
    ],
    "Logistics": [
        ("Rapid Freight AB", "freight delivery and handling"),
        ("North Warehouse", "warehouse storage fee"),
        ("Parcel Link", "courier and parcel shipment"),
    ],
    "Professional Services": [
        ("Berg Legal", "legal advisory engagement"),
        ("Clear Ledger Audit", "accounting and audit support"),
        ("Vector Engineering", "engineering consultancy invoice"),
    ],
}


def _invoice_text(template, invoice):
    amount = f"{invoice['amount']:,.2f}"
    if template == 0:
        return (
            f"INVOICE\nInvoice: {invoice['invoice_number']}\n"
            f"Company: {invoice['company_external_id']}\nVendor: {invoice['vendor_name']}\n"
            f"Issue Date: {invoice['issue_date']}\nDue Date: {invoice['due_date']}\n"
            f"Currency: SEK\nDescription: {invoice['description']}\nTotal: SEK {amount}\n"
        )
    if template == 1:
        return (
            f"Supplier invoice\nInvoice Number: {invoice['invoice_number']}\n"
            f"Customer ID: {invoice['company_external_id']}\nSupplier: {invoice['vendor_name']}\n"
            f"Invoice Date: {invoice['issue_date']}\nPayment Due: {invoice['due_date']}\n"
            f"Service: {invoice['description']}\nAmount Due: SEK {amount}\n"
        )
    return (
        f"PAYABLE DOCUMENT\nReference: {invoice['invoice_number']}\n"
        f"Bill To: {invoice['company_external_id']}\nFrom: {invoice['vendor_name']}\n"
        f"Issued: {invoice['issue_date']}\nDue: {invoice['due_date']}\n"
        f"Details: {invoice['description']}\nInvoice Total: SEK {amount}\n"
    )


def build_dataset(seed=17):
    rng = np.random.default_rng(seed)
    invoices = []
    transactions = []
    transaction_index = 1
    start = date(2026, 1, 5)
    categories = list(CATEGORY_DATA)

    for company_index, (company_id, _name, _industry) in enumerate(COMPANIES):
        for offset in range(18):
            category = categories[(company_index + offset) % len(categories)]
            vendor, description = CATEGORY_DATA[category][offset % 3]
            issue = start + timedelta(days=offset * 8 + company_index)
            due = issue + timedelta(days=30)
            amount = round(float(rng.uniform(1_800, 48_000)), 2)
            number = f"INV-2026-{company_index * 18 + offset + 1:03d}"
            expected = {
                "invoice_number": number,
                "company_external_id": company_id,
                "vendor_name": vendor,
                "issue_date": issue.isoformat(),
                "due_date": due.isoformat(),
                "amount": amount,
                "currency": "SEK",
                "description": description,
                "category": category,
            }
            invoices.append(
                {
                    "document_id": f"DOC-{len(invoices) + 1:04d}",
                    "text": _invoice_text(offset % 3, expected),
                    "expected": expected,
                }
            )

            if offset % 5 != 0:
                booking = due + timedelta(days=int(rng.integers(-3, 6)))
                reference = number if offset % 4 else f"Payment {number}"
                transactions.append(
                    {
                        "transaction_external_id": f"TX-{transaction_index:04d}",
                        "company_external_id": company_id,
                        "booking_date": booking.isoformat(),
                        "amount": -amount,
                        "currency": "SEK",
                        "counterparty": vendor,
                        "reference": reference,
                        "expected_anomaly": "false",
                        "ground_truth_invoice_number": number,
                    }
                )
                transaction_index += 1

    regular_vendors = [item[0] for values in CATEGORY_DATA.values() for item in values]
    for index in range(18):
        company_id = COMPANIES[index % len(COMPANIES)][0]
        amount = round(float(rng.uniform(350, 8_500)), 2)
        transactions.append(
            {
                "transaction_external_id": f"TX-{transaction_index:04d}",
                "company_external_id": company_id,
                "booking_date": (start + timedelta(days=15 + index * 9)).isoformat(),
                "amount": -amount,
                "currency": "SEK",
                "counterparty": regular_vendors[index % len(regular_vendors)],
                "reference": f"Operating expense {index + 1}",
                "expected_anomaly": "false",
                "ground_truth_invoice_number": "",
            }
        )
        transaction_index += 1

    for index in range(8):
        company_id = COMPANIES[index % len(COMPANIES)][0]
        booking = date(2026, 8, 1) + timedelta(days=index * 7)
        amount = float(180_000 + index * 37_500)
        transactions.append(
            {
                "transaction_external_id": f"TX-{transaction_index:04d}",
                "company_external_id": company_id,
                "booking_date": booking.isoformat(),
                "amount": -amount,
                "currency": "SEK",
                "counterparty": f"Unverified Counterparty {index + 1}",
                "reference": f"Manual transfer batch {index + 1}",
                "expected_anomaly": "true",
                "ground_truth_invoice_number": "",
            }
        )
        transaction_index += 1

    return invoices, transactions


def write_dataset(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "companies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["company_external_id", "name", "industry", "country", "currency"],
        )
        writer.writeheader()
        for company_id, name, industry in COMPANIES:
            writer.writerow(
                {
                    "company_external_id": company_id,
                    "name": name,
                    "industry": industry,
                    "country": "SE",
                    "currency": "SEK",
                }
            )

    invoices, transactions = build_dataset()
    with (output / "invoice_documents.jsonl").open("w", encoding="utf-8") as handle:
        for invoice in invoices:
            handle.write(json.dumps(invoice) + "\n")
    with (output / "transactions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transactions[0]))
        writer.writeheader()
        writer.writerows(transactions)
    return {"companies": len(COMPANIES), "invoices": len(invoices), "transactions": len(transactions)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    counts = write_dataset(root / "data" / "demo")
    print(json.dumps(counts, indent=2))
