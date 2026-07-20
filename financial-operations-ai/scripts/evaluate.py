import json
import re
import time
from pathlib import Path

from app.db import SessionLocal
from app.models import Anomaly, Invoice, Reconciliation, Transaction
from app.services.assistant import answer_question
from app.services.ingestion import parse_invoice_document
from scripts.load_demo import load_demo


def classification_metrics(expected, predicted):
    true_positive = sum(a and b for a, b in zip(expected, predicted))
    false_positive = sum(not a and b for a, b in zip(expected, predicted))
    false_negative = sum(a and not b for a, b in zip(expected, predicted))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def extraction_accuracy(data_path):
    correct = 0
    total = 0
    fields = [
        "invoice_number",
        "company_external_id",
        "vendor_name",
        "issue_date",
        "due_date",
        "amount",
        "currency",
        "description",
    ]
    for line in data_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        parsed = parse_invoice_document(record["text"])
        for field in fields:
            expected = record["expected"][field]
            actual = parsed.get(field)
            if field == "amount":
                correct += abs(float(actual) - float(expected)) < 0.01
            else:
                correct += actual == expected
            total += 1
    return correct / total


def run_evaluation():
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    load_demo(reset=True)
    with SessionLocal() as session:
        transactions = session.query(Transaction).all()
        predicted_anomaly_ids = {
            row.transaction_id for row in session.query(Anomaly).all()
        }
        anomaly_metrics = classification_metrics(
            [bool(row.expected_anomaly) for row in transactions],
            [row.id in predicted_anomaly_ids for row in transactions],
        )

        truth_pairs = {
            (row.ground_truth_invoice_number, row.external_id)
            for row in transactions
            if row.ground_truth_invoice_number
        }
        predicted_pairs = set()
        for match in session.query(Reconciliation).all():
            invoice = session.get(Invoice, match.invoice_id)
            transaction = session.get(Transaction, match.transaction_id)
            predicted_pairs.add((invoice.invoice_number, transaction.external_id))
        reconciliation_metrics = classification_metrics(
            [pair in truth_pairs for pair in truth_pairs | predicted_pairs],
            [pair in predicted_pairs for pair in truth_pairs | predicted_pairs],
        )

        invoices = session.query(Invoice).all()
        category_accuracy = sum(
            row.predicted_category == row.expected_category for row in invoices
        ) / len(invoices)

        questions = [
            "Which anomalous payments need investigation?",
            "Show the strongest invoice payment matches.",
            "What is known about INV-2026-011?",
            "What was company revenue in 2024?",
        ]
        citation_scores = []
        refusal_scores = []
        source_use_scores = []
        for question in questions:
            answer, sources, _mode = answer_question(session, question)
            cited = set(re.findall(r"\[((?:invoice|anomaly|reconciliation):\d+)\]", answer))
            available = {source["id"] for source in sources}
            citation_scores.append(float(not cited or cited <= available))
            source_use_scores.append(float(not sources or bool(cited)))
            if not sources:
                refusal_scores.append(float("not contain enough evidence" in answer))

        results = {
            "dataset": {
                "invoices": len(invoices),
                "transactions": len(transactions),
                "labelled_anomalies": sum(bool(row.expected_anomaly) for row in transactions),
                "ground_truth_matches": len(truth_pairs),
            },
            "invoice_extraction_field_accuracy": extraction_accuracy(
                root / "data" / "demo" / "invoice_documents.jsonl"
            ),
            "invoice_category_accuracy": category_accuracy,
            "reconciliation": reconciliation_metrics,
            "anomaly_detection": anomaly_metrics,
            "assistant": {
                "citation_validity": sum(citation_scores) / len(citation_scores),
                "source_use": sum(source_use_scores) / len(source_use_scores),
                "unsupported_question_refusal": sum(refusal_scores) / max(len(refusal_scores), 1),
                "cases": len(questions),
            },
            "runtime_seconds": time.perf_counter() - started,
        }

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = f"""# Evaluation Report

The full pipeline was run on a deterministic synthetic dataset.

| Measure | Result |
| --- | ---: |
| Invoice extraction field accuracy | {results['invoice_extraction_field_accuracy']:.3f} |
| Invoice category accuracy | {results['invoice_category_accuracy']:.3f} |
| Reconciliation precision | {results['reconciliation']['precision']:.3f} |
| Reconciliation recall | {results['reconciliation']['recall']:.3f} |
| Anomaly precision | {results['anomaly_detection']['precision']:.3f} |
| Anomaly recall | {results['anomaly_detection']['recall']:.3f} |
| Assistant citation validity | {results['assistant']['citation_validity']:.3f} |
| Assistant source use | {results['assistant']['source_use']:.3f} |
| Unsupported question refusal | {results['assistant']['unsupported_question_refusal']:.3f} |

Dataset: {results['dataset']['invoices']} invoices and {results['dataset']['transactions']} transactions. The labels contain {results['dataset']['ground_truth_matches']} payment matches and {results['dataset']['labelled_anomalies']} anomalies.

The figures measure this fixed demonstration dataset. They are not estimates of performance on real financial records.
"""
    (reports / "evaluation.md").write_text(markdown, encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))

