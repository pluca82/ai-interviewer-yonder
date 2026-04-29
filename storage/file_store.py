"""Append-only JSON transcripts on disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models.schemas import InterviewTranscript


class FileStore:
    def __init__(self, transcript_dir: Path) -> None:
        self._dir = transcript_dir

    def save_transcript(self, transcript: InterviewTranscript) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        short_id = uuid4().hex[:6]
        path = self._dir / f"interview_{ts}_{short_id}.json"
        path.write_text(
            json.dumps(transcript.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        return path
