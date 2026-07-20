from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db, init_db
from app.models import Anomaly, ApprovalEvent, Company, Invoice, Reconciliation, Transaction
from app.schemas import ApprovalRequest, AssistantRequest, AssistantResponse
from app.services.assistant import answer_question
from app.services.ingestion import import_companies, import_invoice_documents, import_transactions
from app.services.pipeline import run_analysis


ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title="Financial Operations AI", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def interface():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/import/companies")
async def upload_companies(file: UploadFile = File(...), session: Session = Depends(get_db)):
    return {"imported": import_companies(session, await file.read())}


@app.post("/api/import/invoices")
async def upload_invoices(file: UploadFile = File(...), session: Session = Depends(get_db)):
    return {"imported": import_invoice_documents(session, await file.read())}


@app.post("/api/import/transactions")
async def upload_transactions(file: UploadFile = File(...), session: Session = Depends(get_db)):
    return {"imported": import_transactions(session, await file.read())}


@app.post("/api/process")
def process(session: Session = Depends(get_db)):
    if session.query(Invoice).count() == 0 or session.query(Transaction).count() == 0:
        raise HTTPException(400, "Import invoices and transactions before running analysis.")
    return run_analysis(session)


@app.get("/api/summary")
def summary(session: Session = Depends(get_db)):
    total_invoice_value = session.query(func.coalesce(func.sum(Invoice.amount), 0)).scalar()
    return {
        "companies": session.query(Company).count(),
        "invoices": session.query(Invoice).count(),
        "transactions": session.query(Transaction).count(),
        "suggested_matches": session.query(Reconciliation).filter_by(status="suggested").count(),
        "approved_matches": session.query(Reconciliation).filter_by(status="approved").count(),
        "open_anomalies": session.query(Anomaly).filter_by(status="open").count(),
        "invoice_value": round(float(total_invoice_value), 2),
    }


@app.get("/api/reconciliations")
def reconciliations(session: Session = Depends(get_db)):
    rows = (
        session.query(Reconciliation, Invoice, Transaction)
        .join(Invoice, Reconciliation.invoice_id == Invoice.id)
        .join(Transaction, Reconciliation.transaction_id == Transaction.id)
        .order_by(Reconciliation.score.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": match.id,
            "invoice": invoice.invoice_number,
            "transaction": transaction.external_id,
            "vendor": invoice.vendor_name,
            "amount": invoice.amount,
            "currency": invoice.currency,
            "score": round(match.score, 3),
            "status": match.status,
            "rationale": match.rationale,
        }
        for match, invoice, transaction in rows
    ]


@app.post("/api/reconciliations/{match_id}/approve")
def approve_reconciliation(
    match_id: int, request: ApprovalRequest, session: Session = Depends(get_db)
):
    match = session.get(Reconciliation, match_id)
    if not match:
        raise HTTPException(404, "Reconciliation not found.")
    if match.status == "approved":
        raise HTTPException(409, "Reconciliation is already approved.")
    match.status = "approved"
    session.add(
        ApprovalEvent(
            entity_type="reconciliation",
            entity_id=match.id,
            action="approved",
            reviewer=request.reviewer,
            note=request.note,
        )
    )
    session.commit()
    return {"id": match.id, "status": match.status, "reviewer": request.reviewer}


@app.get("/api/anomalies")
def anomalies(session: Session = Depends(get_db)):
    rows = (
        session.query(Anomaly, Transaction, Company)
        .join(Transaction, Anomaly.transaction_id == Transaction.id)
        .join(Company, Transaction.company_id == Company.id)
        .order_by(Anomaly.score.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": anomaly.id,
            "transaction": transaction.external_id,
            "company": company.name,
            "counterparty": transaction.counterparty,
            "amount": abs(transaction.amount),
            "currency": transaction.currency,
            "score": round(anomaly.score, 3),
            "severity": anomaly.severity,
            "reasons": anomaly.reasons,
            "status": anomaly.status,
        }
        for anomaly, transaction, company in rows
    ]


@app.post("/api/assistant", response_model=AssistantResponse)
def assistant(request: AssistantRequest, session: Session = Depends(get_db)):
    answer, sources, mode = answer_question(session, request.question)
    return AssistantResponse(answer=answer, sources=sources, mode=mode)

