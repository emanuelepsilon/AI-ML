import json

from app.services.ingestion import parse_invoice_document


def test_all_invoice_templates_extract_required_fields(demo_path):
    records = [
        json.loads(line)
        for line in (demo_path / "invoice_documents.jsonl").read_text().splitlines()
    ]
    required = {
        "invoice_number",
        "company_external_id",
        "vendor_name",
        "issue_date",
        "due_date",
        "amount",
        "currency",
        "description",
    }
    for record in records:
        parsed = parse_invoice_document(record["text"])
        assert required <= parsed.keys()
        assert parsed["invoice_number"] == record["expected"]["invoice_number"]
        assert parsed["amount"] == record["expected"]["amount"]


def test_invoice_classifier_uses_model_output(loaded_session):
    from app.models import Invoice

    invoices = loaded_session.query(Invoice).all()
    accuracy = sum(row.predicted_category == row.expected_category for row in invoices) / len(
        invoices
    )
    assert accuracy >= 0.9

