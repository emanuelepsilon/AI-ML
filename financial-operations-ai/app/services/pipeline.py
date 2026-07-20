from sqlalchemy.orm import Session

from app.services.anomaly import detect_anomalies
from app.services.reconciliation import reconcile


def run_analysis(session: Session):
    matches = reconcile(session)
    anomalies = detect_anomalies(session)
    return {"suggested_matches": len(matches), "open_anomalies": len(anomalies)}
