from app.models import Anomaly, Invoice, Reconciliation, Transaction


def test_reconciliation_recovers_labelled_pairs(loaded_session):
    truth = {
        (row.ground_truth_invoice_number, row.external_id)
        for row in loaded_session.query(Transaction).all()
        if row.ground_truth_invoice_number
    }
    predicted = set()
    for match in loaded_session.query(Reconciliation).all():
        invoice = loaded_session.get(Invoice, match.invoice_id)
        transaction = loaded_session.get(Transaction, match.transaction_id)
        predicted.add((invoice.invoice_number, transaction.external_id))
        assert match.status == "suggested"
    assert len(predicted & truth) / len(truth) >= 0.95
    assert len(predicted & truth) / len(predicted) >= 0.95


def test_anomaly_detector_finds_labelled_outliers(loaded_session):
    transactions = loaded_session.query(Transaction).all()
    expected = {row.id for row in transactions if row.expected_anomaly}
    predicted = {row.transaction_id for row in loaded_session.query(Anomaly).all()}
    assert len(expected & predicted) / len(expected) >= 0.8
    assert len(expected & predicted) / len(predicted) >= 0.6

