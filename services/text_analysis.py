"""Deterministic bonus analysis: keyword frequencies only (no extra deps)."""

from __future__ import annotations

import re
from collections import Counter

from models.schemas import BonusAnalysis, KeywordStat, QAPair

_TOKEN_RE = re.compile(r"[a-z][a-z0-9']{2,}", re.IGNORECASE)

_STOPWORDS = frozenset(
    """
    the a an and or but if to of in on for with as by at from into through during
    before after above below between under again further then once here there when
    where why how all each both few more most other some such only own same so than
    too very can will just don should now use one also like get make way may using
    used uses using well would could about your our their them they you we it its
    this that these those was were been being have has had does did doing done
    what which who whom whose into out over such any than then them than
    """.split()
)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _keyword_stats(corpus: str, top_n: int = 20) -> list[KeywordStat]:
    counts: Counter[str] = Counter()
    for tok in _tokens(corpus):
        if tok in _STOPWORDS:
            continue
        counts[tok] += 1
    return [KeywordStat(term=w, count=c) for w, c in counts.most_common(top_n)]


def build_bonus_analysis(topic: str, qa_pairs: list[QAPair]) -> BonusAnalysis:
    parts = [topic.lower()]
    for qa in qa_pairs:
        parts.append(qa.answer.lower())
        parts.append(qa.question_text.lower())
    corpus = " ".join(parts)
    return BonusAnalysis(keywords=_keyword_stats(corpus))
