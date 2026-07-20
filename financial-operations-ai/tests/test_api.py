from fastapi.testclient import TestClient

from app.main import app
from app.models import ApprovalEvent, Reconciliation


def test_match_requires_named_human_approval(loaded_session):
    match = loaded_session.query(Reconciliation).first()
    with TestClient(app) as client:
        invalid = client.post(
            f"/api/reconciliations/{match.id}/approve", json={"reviewer": "x"}
        )
        assert invalid.status_code == 422
        approved = client.post(
            f"/api/reconciliations/{match.id}/approve",
            json={"reviewer": "Emanuel Melki", "note": "Invoice and payment checked"},
        )
        assert approved.status_code == 200
    loaded_session.expire_all()
    assert loaded_session.get(Reconciliation, match.id).status == "approved"
    event = loaded_session.query(ApprovalEvent).filter_by(entity_id=match.id).one()
    assert event.reviewer == "Emanuel Melki"

