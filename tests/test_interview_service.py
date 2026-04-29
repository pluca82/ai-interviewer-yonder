from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm.client import LLMClient
from llm.prompts import PromptBuilder
from models.schemas import NextQuestionLLMOut, QAPair
from services.interview_service import INTERVIEW_ALREADY_COMPLETE_DETAIL, InterviewService
from storage.file_store import FileStore


def _pairs(n: int) -> list[QAPair]:
    return [
        QAPair(
            question_id=f"q{i}",
            question_text=f"Question {i}?",
            answer=f"Answer {i}.",
        )
        for i in range(1, n + 1)
    ]


def _svc(llm: object, tmp_path: Path) -> InterviewService:
    return InterviewService(llm, PromptBuilder(), FileStore(tmp_path))


def test_stop_after_three_when_model_completes(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.chat_json.return_value = {"interview_complete": True, "question": None}
    svc = _svc(llm, tmp_path)
    out = svc.generate_next_question("topic", _pairs(3))
    assert out.interview_complete is True
    assert out.question is None
    llm.chat_json.assert_called_once()


def test_four_pairs_returns_fifth_question(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.chat_json.return_value = {
        "interview_complete": False,
        "question": {
            "id": "q5",
            "text": "Final question?",
            "intent": "wrap-up",
            "difficulty": "hard",
        },
    }
    svc = _svc(llm, tmp_path)
    out = svc.generate_next_question("topic", _pairs(4))
    assert out.interview_complete is False
    assert out.question is not None
    assert out.question.id == "q5"
    assert out.question.text == "Final question?"


def test_five_pairs_raises_without_llm(tmp_path: Path) -> None:
    llm = MagicMock()
    svc = _svc(llm, tmp_path)
    with pytest.raises(ValueError, match=INTERVIEW_ALREADY_COMPLETE_DETAIL):
        svc.generate_next_question("topic", _pairs(5))
    llm.chat_json.assert_not_called()


def test_invalid_llm_shape_raises_value_error(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.chat_json.return_value = {"wrong": True}
    svc = _svc(llm, tmp_path)
    with pytest.raises(ValueError, match="does not match the next-question schema"):
        svc.generate_next_question("topic", [])


def test_chat_json_valid_json_wrong_schema_then_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llm.client.OpenAI", lambda **kw: MagicMock())
    settings = MagicMock(
        openai_model="m",
        json_response_format=False,
        openai_api_key="k",
        openai_base_url=None,
    )
    client = LLMClient(settings)
    payloads = [
        '{"completed": true, "extra": "oops"}',
        '{"interview_complete": true, "question": null}',
    ]
    n = [0]

    def fake_raw(self: LLMClient, messages: list[dict[str, str]]) -> str:
        i = n[0]
        n[0] += 1
        return payloads[i]

    monkeypatch.setattr(LLMClient, "_completion_raw", fake_raw)
    out = client.chat_json(
        [{"role": "user", "content": "hi"}],
        schema=NextQuestionLLMOut,
    )
    assert out["interview_complete"] is True
    assert n[0] == 2


def test_chat_json_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm.client.OpenAI", lambda **kw: MagicMock())
    settings = MagicMock(
        openai_model="m",
        json_response_format=False,
        openai_api_key="k",
        openai_base_url=None,
    )
    client = LLMClient(settings)
    calls: list[str] = []

    def fake_raw(self: LLMClient, messages: list[dict[str, str]]) -> str:
        calls.append(str(len(messages)))
        if len(calls) == 1:
            return "not json {"
        return '{"interview_complete": true, "question": null}'

    monkeypatch.setattr(LLMClient, "_completion_raw", fake_raw)
    out = client.chat_json([{"role": "user", "content": "hi"}], schema=NextQuestionLLMOut)
    assert out["interview_complete"] is True
    assert len(calls) == 2


def test_chat_json_raises_after_two_failed_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm.client.OpenAI", lambda **kw: MagicMock())
    settings = MagicMock(
        openai_model="m",
        json_response_format=False,
        openai_api_key="k",
        openai_base_url=None,
    )
    client = LLMClient(settings)

    def fake_raw(self: LLMClient, messages: list[dict[str, str]]) -> str:
        return "@@@"

    monkeypatch.setattr(LLMClient, "_completion_raw", fake_raw)
    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        client.chat_json([{"role": "user", "content": "hi"}])
