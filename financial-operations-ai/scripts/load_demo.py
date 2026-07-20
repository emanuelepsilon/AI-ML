from pathlib import Path

from app import db
from app.services.ingestion import import_companies, import_invoice_documents, import_transactions
from app.services.pipeline import run_analysis


def load_demo(reset=True):
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "demo"
    db.init_db()
    if reset:
        db.reset_db()
    with db.SessionLocal() as session:
        imported = {
            "companies": import_companies(session, (data / "companies.csv").read_bytes()),
            "invoices": import_invoice_documents(
                session, (data / "invoice_documents.jsonl").read_bytes()
            ),
            "transactions": import_transactions(
                session, (data / "transactions.csv").read_bytes()
            ),
        }
        imported.update(run_analysis(session))
    return imported


if __name__ == "__main__":
    print(load_demo())
