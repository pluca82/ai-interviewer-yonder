from __future__ import annotations

from pathlib import Path

from llm.client import LLMClient
from llm.prompts import PromptBuilder
from models.schemas import (
    NextQuestionLLMOut,
    NextQuestionResponse,
    QAPair,
    QuestionItem,
    SummaryOut,
    SummaryResponse,
    InterviewTranscript,
)
from pydantic import ValidationError
from services.text_analysis import build_bonus_analysis
from storage.file_store import FileStore

INTERVIEW_ALREADY_COMPLETE_DETAIL = "Interview already complete; use summary."


class InterviewService:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptBuilder,
        file_store: FileStore,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._file_store = file_store

    def generate_next_question(
        self,
        topic: str,
        qa_pairs: list[QAPair],
        questions: list[QuestionItem] | None = None,
    ) -> NextQuestionResponse:
        topic_clean = topic.strip()
        n = len(qa_pairs)
        if n >= 5:
            raise ValueError(INTERVIEW_ALREADY_COMPLETE_DETAIL)

        messages = self._prompts.build_next_question_messages(topic_clean, qa_pairs, questions)
        raw = self._llm.chat_json(messages, schema=NextQuestionLLMOut)
        try:
            out = NextQuestionLLMOut.model_validate(raw)
        except ValidationError as e:
            raise ValueError("Model returned JSON that does not match the next-question schema") from e

        if n < 3 and (out.interview_complete or out.question is None):
            raise ValueError("Model ended the interview before three answered rounds")
        if n == 4 and (out.interview_complete or out.question is None):
            raise ValueError("Model must return the fifth question after four answers")

        if out.interview_complete:
            return NextQuestionResponse(
                topic=topic_clean,
                interview_complete=True,
                question=None,
                round_number=n,
            )

        if out.question is None:
            raise ValueError("Model returned no question while interview_complete was false")

        assigned_id = f"q{n + 1}"
        q = out.question.model_copy(update={"id": assigned_id})
        return NextQuestionResponse(
            topic=topic_clean,
            interview_complete=False,
            question=q,
            round_number=n + 1,
        )

    def generate_summary(
        self,
        topic: str,
        qa_pairs: list[QAPair],
        questions: list[QuestionItem] | None = None,
    ) -> tuple[SummaryResponse, Path]:
        topic_clean = topic.strip()
        messages = self._prompts.build_summary_messages(topic_clean, qa_pairs, questions)
        raw = self._llm.chat_json(messages, schema=SummaryOut)
        try:
            summary = SummaryOut.model_validate(raw)
        except ValidationError as e:
            raise ValueError("Model returned JSON that does not match the summary schema") from e
        bonus = build_bonus_analysis(topic_clean, qa_pairs)
        transcript = InterviewTranscript(
            topic=topic_clean,
            questions=list(questions or []),
            qa_pairs=qa_pairs,
            summary=summary,
            bonus_analysis=bonus,
        )
        path = self._file_store.save_transcript(transcript)
        return (
            SummaryResponse(topic=topic_clean, summary=summary, bonus_analysis=bonus),
            path,
        )
