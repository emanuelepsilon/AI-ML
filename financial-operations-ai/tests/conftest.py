from pathlib import Path

import pytest

from app import db
from app.services.ingestion import import_companies, import_invoice_documents, import_transactions
from app.services.pipeline import run_analysis
from scripts.generate_demo_data import write_dataset


@pytest.fixture()
def loaded_session(tmp_path):
    db.configure_database("sqlite+pysqlite:///:memory:")
    db.init_db()
    data = tmp_path / "demo"
    write_dataset(data)
    with db.SessionLocal() as session:
        import_companies(session, (data / "companies.csv").read_bytes())
        import_invoice_documents(session, (data / "invoice_documents.jsonl").read_bytes())
        import_transactions(session, (data / "transactions.csv").read_bytes())
        run_analysis(session)
        yield session


@pytest.fixture()
def demo_path(tmp_path):
    path = Path(tmp_path) / "demo"
    write_dataset(path)
    return path

