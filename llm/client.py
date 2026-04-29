"""Thin OpenAI wrapper — prompts live elsewhere."""

from __future__ import annotations

import json

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from core.config import Settings


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_object(raw: str) -> dict:
    text = _strip_markdown_fences(raw)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        out = json.loads(text[start : end + 1])
    if not isinstance(out, dict):
        raise ValueError("LLM JSON root must be an object")
    return out


_JSON_REPAIR_USER = (
    "Return ONLY valid JSON matching the expected schema. No explanation."
)


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.openai_model
        self._json_response_format = settings.json_response_format
        kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)

    def _completion_raw(self, messages: list[dict[str, str]]) -> str:
        params: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.4,
        }
        if self._json_response_format:
            params["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**params)
        return (resp.choices[0].message.content or "").strip()

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: type[BaseModel] | None = None,
    ) -> dict:
        def parse_attempt(raw: str) -> dict:
            text = (raw or "").strip()
            if not text:
                raise ValueError("empty response")
            data = _parse_json_object(text)
            if schema is not None:
                schema.model_validate(data)
            return data

        try:
            return parse_attempt(self._completion_raw(messages))
        except (json.JSONDecodeError, ValueError, ValidationError):
            retry_messages = [
                *messages,
                {"role": "user", "content": _JSON_REPAIR_USER},
            ]
            try:
                return parse_attempt(self._completion_raw(retry_messages))
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                raise ValueError("LLM returned invalid JSON") from e
