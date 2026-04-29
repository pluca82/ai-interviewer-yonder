"""Request/response shapes and transcript model for the interview simulator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Difficulty = Literal["easy", "medium", "hard"]
SentimentLabel = Literal["positive", "neutral", "negative"]
ConfidenceLevel = Literal["low", "medium", "high"]


class QuestionItem(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    intent: str = Field(
        min_length=1,
        description="What this question is trying to evaluate.",
    )
    difficulty: Difficulty


class QAPair(BaseModel):
    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class NextQuestionRequest(BaseModel):
    """Stateless: client sends full history each time."""

    topic: str = Field(min_length=1, max_length=500)
    qa_pairs: list[QAPair] = Field(default_factory=list)
    questions: list[QuestionItem] | None = None

    @field_validator("qa_pairs")
    @classmethod
    def _not_huge(cls, v: list[QAPair]) -> list[QAPair]:
        if len(v) > 20:
            raise ValueError("Too many Q&A pairs")
        return v


class NextQuestionResponse(BaseModel):
    topic: str
    interview_complete: bool
    question: QuestionItem | None = None
    """1-based index of the question just returned, or last completed count if interview_complete."""

    round_number: int = Field(ge=0, le=5)


class NextQuestionLLMOut(BaseModel):
    """Root JSON from the LLM for one adaptive step."""

    interview_complete: bool
    question: QuestionItem | None = None


class SummaryRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    qa_pairs: list[QAPair] = Field(min_length=1)
    questions: list[QuestionItem] | None = None

    @field_validator("qa_pairs")
    @classmethod
    def _not_huge(cls, v: list[QAPair]) -> list[QAPair]:
        if len(v) > 20:
            raise ValueError("Too many Q&A pairs")
        return v


class SentimentBlock(BaseModel):
    label: SentimentLabel
    explanation: str = Field(min_length=1)


class SummaryOut(BaseModel):
    """Structured debrief returned by the LLM."""

    confidence: ConfidenceLevel
    key_themes: list[str]
    sentiment: SentimentBlock
    strengths: list[str]
    areas_for_improvement: list[str]
    notable_points: list[str]


class KeywordStat(BaseModel):
    """Deterministic keyword frequency (bonus analysis)."""

    term: str
    count: int = Field(ge=1)


class BonusAnalysis(BaseModel):
    """Non-LLM extras; sentiment stays in `summary.sentiment` only."""

    keywords: list[KeywordStat]


class SummaryResponse(BaseModel):
    topic: str
    summary: SummaryOut
    bonus_analysis: BonusAnalysis


class InterviewTranscript(BaseModel):
    """Written to disk after a successful summary (JSON file)."""

    topic: str
    questions: list[QuestionItem]
    qa_pairs: list[QAPair]
    summary: SummaryOut
    bonus_analysis: BonusAnalysis
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
