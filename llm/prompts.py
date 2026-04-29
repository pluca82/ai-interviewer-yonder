"""System and user messages for the LLM — no HTTP here."""

from __future__ import annotations

import json

from models.schemas import QAPair, QuestionItem


class PromptBuilder:
    def build_next_question_messages(
        self,
        topic: str,
        qa_pairs: list[QAPair],
        questions: list[QuestionItem] | None,
    ) -> list[dict[str, str]]:
        completed = len(qa_pairs)
        schema_hint = json.dumps(
            {
                "interview_complete": False,
                "question": {
                    "id": "qN (ignored; server assigns)",
                    "text": "string",
                    "intent": "what this question evaluates",
                    "difficulty": "easy | medium | hard",
                },
            },
            indent=2,
        )
        schema_alt = (
            'When you end the interview (see stopping rules below), return exactly:\n'
            '{"interview_complete": true, "question": null}\n'
            "You must NOT do this before 3 full Q&A pairs exist, and you must NOT do it when there are already "
            "4 answered pairs (the fifth question is required next)."
        )
        system = (
            "You are a professional technical interviewer conducting one question at a time. "
            "Tone: neutral, respectful, concise. Verbosity: tight question text. "
            "Return ONLY valid JSON. No markdown fences, no commentary outside the JSON object.\n\n"
            f"Shape (when asking another question):\n{schema_hint}\n\n"
            f"{schema_alt}\n\n"
            "Rules:\n"
            "- You receive the topic and everything said so far (prior Q&A). Output **exactly one** JSON object: "
            "either end the interview (interview_complete true, question null) or ask **one** next question.\n"
            "- If you ask another question: it may be a follow-up on the last answer or a new angle — but only "
            "after you have decided continuing is justified (see stopping rules).\n"
            "- Difficulty must be one of: easy, medium, hard — chosen according to this progression: "
            "round 1 → easy or medium; round 2 → medium; round 3+ → medium or hard. "
            "Never go back to easy after round 1. The interview must feel progressively more demanding.\n"
            f"- Completed Q&A pairs so far: {completed}. Maximum 5 questions total.\n"
            "\n"
            "Stopping (read carefully):\n"
            "- If completed is 0, 1, or 2: you MUST return interview_complete false and a non-null question. "
            "Do not end the interview early.\n"
            "- If completed is 3 (three full answers already in the transcript): you SHOULD end the interview "
            "with interview_complete true and question null **unless** the candidate's last answer raised a "
            "**specific, concrete** unresolved point that genuinely requires **one** more question to evaluate "
            "fairly. Default to stopping. \"More depth is possible\" or \"another angle exists\" is **not** a "
            "sufficient reason to continue — only a clear gap or ambiguity that one targeted question would resolve.\n"
            "- If completed is 4: you MUST return interview_complete false and the fifth question (question non-null). "
            "Never end here; the fifth question is mandatory.\n"
            "- Tiebreaker: if you are unsure whether to stop or continue, **stop**. Prefer "
            '{"interview_complete": true, "question": null} over asking another question whenever judgment is close.\n'
            "- Vague topics (e.g. 'test'): still produce reasonable screening questions in early rounds; "
            "stopping rules still apply once completed reaches 3.\n"
        )
        lines = [f'Interview topic: "{topic}"', "", "Conversation so far:"]
        if questions:
            lines.append("(Prior question intents)")
            for q in questions:
                lines.append(f"- [{q.id}] {q.intent} ({q.difficulty})")
            lines.append("")
        if not qa_pairs:
            lines.append("(No answers yet — produce the opening question.)")
        else:
            for i, qa in enumerate(qa_pairs, start=1):
                lines.append(f"Q{i} (id={qa.question_id}): {qa.question_text}")
                lines.append(f"A{i}: {qa.answer}")
                lines.append("")
        user = "\n".join(lines).strip() + "\n\nReturn the JSON for this single step now."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def build_summary_messages(
        self,
        topic: str,
        qa_pairs: list[QAPair],
        questions: list[QuestionItem] | None = None,
    ) -> list[dict[str, str]]:
        schema_hint = json.dumps(
            {
                "confidence": "low | medium | high",
                "key_themes": ["string"],
                "sentiment": {"label": "positive | neutral | negative", "explanation": "string"},
                "strengths": ["string"],
                "areas_for_improvement": ["string"],
                "notable_points": ["string"],
            },
            indent=2,
        )
        system = (
            "You debrief a mock interview from the candidate's written answers. "
            "Tone: constructive and professional. Verbosity: concise bullets and short sentences. "
            "Return ONLY valid JSON matching the shape below. "
            "No markdown fences, no commentary, no text outside the JSON object.\n\n"
            f"Required JSON shape:\n{schema_hint}\n\n"
            "Rules:\n"
            "- 'confidence' reflects how reliable your debrief is given answer length, vagueness, or contradictions "
            "(low if very little to go on).\n"
            "- Arrays may be empty only if nothing applies; prefer at least one item when evidence exists.\n"
            "- sentiment.label must be exactly one of: positive, neutral, negative.\n"
        )
        lines = [f'Topic: "{topic}"', "", "Q&A transcript:"]
        if questions:
            lines.append("(Original question intents for context)")
            for q in questions:
                lines.append(f"- [{q.id}] {q.intent} ({q.difficulty})")
            lines.append("")
        for i, qa in enumerate(qa_pairs, start=1):
            lines.append(f"Q{i} (id={qa.question_id}): {qa.question_text}")
            lines.append(f"A{i}: {qa.answer}")
            lines.append("")
        user = "\n".join(lines).strip() + "\n\nProduce the summary JSON now."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
