from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from openai import APIStatusError, AuthenticationError, OpenAIError

from core.config import get_settings
from llm.client import LLMClient
from llm.prompts import PromptBuilder
from models.schemas import NextQuestionRequest, NextQuestionResponse, SummaryRequest, SummaryResponse
from services.interview_service import INTERVIEW_ALREADY_COMPLETE_DETAIL, InterviewService
from storage.file_store import FileStore

router = APIRouter(tags=["interview"])


def _llm_http_detail(exc: OpenAIError) -> str:
    if isinstance(exc, AuthenticationError):
        return (
            "LLM API rejected the credentials (401). Check OPENAI_API_KEY in .env — "
            "Groq, OpenAI, and other OpenAI-compatible providers all use that name — "
            "no quotes around the value, no stray spaces. Restart the app after editing .env."
        )
    if isinstance(exc, APIStatusError) and exc.status_code == 401:
        return (
            "401 Unauthorized from your LLM endpoint. Confirm the key matches the "
            "provider in OPENAI_BASE_URL (leave it unset for OpenAI’s default). "
            "Then restart the app."
        )
    return "LLM request failed"


@lru_cache
def _interview_service() -> InterviewService:
    settings = get_settings()
    return InterviewService(
        LLMClient(settings),
        PromptBuilder(),
        FileStore(settings.transcript_dir),
    )


def interview_service_dep() -> InterviewService:
    return _interview_service()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/interview/question/next", response_model=NextQuestionResponse)
def post_next_question(
    body: NextQuestionRequest,
    svc: InterviewService = Depends(interview_service_dep),
) -> NextQuestionResponse:
    try:
        return svc.generate_next_question(body.topic, body.qa_pairs, body.questions)
    except ValueError as e:
        if str(e) == INTERVIEW_ALREADY_COMPLETE_DETAIL:
            raise HTTPException(status_code=400, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=_llm_http_detail(e)) from e


@router.post("/interview/summary", response_model=SummaryResponse)
def post_summary(
    body: SummaryRequest,
    svc: InterviewService = Depends(interview_service_dep),
) -> SummaryResponse:
    try:
        response, _path = svc.generate_summary(
            body.topic,
            body.qa_pairs,
            body.questions,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=_llm_http_detail(e)) from e
