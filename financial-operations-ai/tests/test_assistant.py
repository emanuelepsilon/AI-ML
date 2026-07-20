import re

from app.services.assistant import answer_question


def test_supported_answer_uses_valid_sources(loaded_session):
    answer, sources, mode = answer_question(
        loaded_session, "Which anomalous payments need investigation?"
    )
    available = {source["id"] for source in sources}
    cited = set(re.findall(r"\[((?:anomaly|invoice|reconciliation):\d+)\]", answer))
    assert mode == "deterministic"
    assert sources
    assert cited
    assert cited <= available


def test_unsupported_answer_refuses(loaded_session):
    answer, sources, _mode = answer_question(
        loaded_session, "What was company revenue in 2024?"
    )
    assert sources == []
    assert "not contain enough evidence" in answer

