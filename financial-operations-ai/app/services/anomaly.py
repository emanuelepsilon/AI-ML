import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy.orm import Session

from app.models import Anomaly, Transaction


def _features(transactions):
    amounts = np.array([abs(item.amount) for item in transactions], dtype=float)
    log_amounts = np.log1p(amounts)
    features = []
    for item, amount, log_amount in zip(transactions, amounts, log_amounts):
        features.append(
            [
                log_amount,
                float(item.booking_date.weekday() >= 5),
                float(amount % 1000 == 0),
                len(item.reference),
            ]
        )
    return np.asarray(features), amounts


def detect_anomalies(session: Session, contamination=0.08):
    session.query(Anomaly).delete()
    transactions = session.query(Transaction).order_by(Transaction.id).all()
    if len(transactions) < 5:
        session.commit()
        return []

    features, amounts = _features(transactions)
    model = IsolationForest(contamination=contamination, random_state=17, n_estimators=160)
    labels = model.fit_predict(features)
    raw_scores = -model.decision_function(features)
    amount_cutoff = np.quantile(amounts, 0.97)
    known_counterparties = {
        item.counterparty.lower()
        for item in transactions
        if item.ground_truth_invoice_number
    }

    anomalies = []
    for item, label, raw_score, amount in zip(transactions, labels, raw_scores, amounts):
        reasons = []
        if label == -1:
            reasons.append("Isolation Forest outlier")
        if amount >= amount_cutoff:
            reasons.append("Amount above the 97th percentile")
        if item.counterparty.lower() not in known_counterparties and amount > np.median(amounts) * 2:
            reasons.append("Large payment to an unfamiliar counterparty")
        if item.booking_date.weekday() >= 5 and amount > np.median(amounts) * 3:
            reasons.append("Large weekend payment")
        if not reasons:
            continue
        score = float(min(1.0, 0.55 + max(raw_score, 0.0) * 3 + 0.1 * len(reasons)))
        anomaly = Anomaly(
            transaction_id=item.id,
            score=score,
            severity="high" if score >= 0.82 else "medium",
            reasons="; ".join(reasons),
            status="open",
        )
        session.add(anomaly)
        anomalies.append(anomaly)
    session.commit()
    return anomalies
