from difflib import SequenceMatcher

from sqlalchemy.orm import Session

from app.models import Invoice, Reconciliation, Transaction


def _similarity(left, right):
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()


def score_pair(invoice, transaction):
    amount_difference = abs(invoice.amount - abs(transaction.amount))
    amount_score = max(0.0, 1.0 - amount_difference / max(invoice.amount, 1.0))
    days = abs((transaction.booking_date - invoice.due_date).days)
    date_score = max(0.0, 1.0 - days / 30.0)
    reference_score = max(
        1.0 if invoice.invoice_number.lower() in transaction.reference.lower() else 0.0,
        _similarity(invoice.vendor_name, transaction.counterparty),
    )
    currency_score = 1.0 if invoice.currency == transaction.currency else 0.0
    total = 0.48 * amount_score + 0.20 * date_score + 0.27 * reference_score + 0.05 * currency_score
    return total, amount_score, date_score, reference_score


def reconcile(session: Session, threshold=0.72):
    session.query(Reconciliation).delete()
    invoices = session.query(Invoice).all()
    transactions = session.query(Transaction).all()
    candidates = []
    for invoice in invoices:
        for transaction in transactions:
            if invoice.company_id != transaction.company_id:
                continue
            score, amount_score, date_score, reference_score = score_pair(invoice, transaction)
            if score >= threshold:
                candidates.append(
                    (
                        score,
                        invoice,
                        transaction,
                        amount_score,
                        date_score,
                        reference_score,
                    )
                )

    used_invoices = set()
    used_transactions = set()
    matches = []
    for score, invoice, transaction, amount_score, date_score, reference_score in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        if invoice.id in used_invoices or transaction.id in used_transactions:
            continue
        used_invoices.add(invoice.id)
        used_transactions.add(transaction.id)
        match = Reconciliation(
            invoice_id=invoice.id,
            transaction_id=transaction.id,
            score=score,
            amount_score=amount_score,
            date_score=date_score,
            reference_score=reference_score,
            status="suggested",
            rationale=(
                f"Amount {amount_score:.2f}, date {date_score:.2f}, "
                f"reference {reference_score:.2f}. Human approval required."
            ),
        )
        session.add(match)
        matches.append(match)
    session.commit()
    return matches

