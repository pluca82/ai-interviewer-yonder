from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.routes import interview_service_dep
from main import app
from services.interview_service import INTERVIEW_ALREADY_COMPLETE_DETAIL


@pytest.fixture
def client(mock_svc: MagicMock) -> TestClient:
    app.dependency_overrides[interview_service_dep] = lambda: mock_svc
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def mock_svc() -> MagicMock:
    return MagicMock()


def test_next_question_invalid_schema_returns_502(client: TestClient, mock_svc: MagicMock) -> None:
    mock_svc.generate_next_question.side_effect = ValueError(
        "Model returned JSON that does not match the next-question schema",
    )
    res = client.post(
        "/interview/question/next",
        json={"topic": "x", "qa_pairs": []},
    )
    assert res.status_code == 502
    assert "does not match" in res.json()["detail"]


def test_next_question_five_pairs_returns_400(client: TestClient, mock_svc: MagicMock) -> None:
    mock_svc.generate_next_question.side_effect = ValueError(INTERVIEW_ALREADY_COMPLETE_DETAIL)
    res = client.post(
        "/interview/question/next",
        json={"topic": "x", "qa_pairs": []},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == INTERVIEW_ALREADY_COMPLETE_DETAIL
