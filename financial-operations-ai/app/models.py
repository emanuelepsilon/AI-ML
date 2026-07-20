from datetime import UTC, date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    industry: Mapped[str] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(3))
    currency: Mapped[str] = mapped_column(String(3))


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(primary_key=True)
    invoice_number: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    vendor_name: Mapped[str] = mapped_column(String(160), index=True)
    issue_date: Mapped[date] = mapped_column(Date)
    due_date: Mapped[date] = mapped_column(Date)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    description: Mapped[str] = mapped_column(Text)
    predicted_category: Mapped[str] = mapped_column(String(80))
    expected_category: Mapped[str | None] = mapped_column(String(80), nullable=True)
    raw_document: Mapped[str] = mapped_column(Text)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    booking_date: Mapped[date] = mapped_column(Date)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    counterparty: Mapped[str] = mapped_column(String(160), index=True)
    reference: Mapped[str] = mapped_column(String(200))
    expected_anomaly: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ground_truth_invoice_number: Mapped[str | None] = mapped_column(String(60), nullable=True)


class Reconciliation(Base):
    __tablename__ = "reconciliations"

    id: Mapped[int] = mapped_column(primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), index=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"), index=True)
    score: Mapped[float] = mapped_column(Float)
    amount_score: Mapped[float] = mapped_column(Float)
    date_score: Mapped[float] = mapped_column(Float)
    reference_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), default="suggested")
    rationale: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"), index=True)
    score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(20))
    reasons: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="open")


class ApprovalEvent(Base):
    __tablename__ = "approval_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[int] = mapped_column(index=True)
    action: Mapped[str] = mapped_column(String(40))
    reviewer: Mapped[str] = mapped_column(String(120))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
