"""§5.3 / §5.4 — concrete scorers (drop-in implementations of the ABCs).

Phase 1b ships the LEXICON fallback scorers (no model download, CI-friendly).
The dense backends — KoELECTRA sentiment / 5-emotion sum (§5.3) and zero-shot NLI
entailment (§5.4) — implement the same `score(text)->[0,1]` interface and swap in
later without touching the pipelines (§4 design principle).

⚠️ Sparse lexicon branching is the FALLBACK only; the design target is a dense,
continuous score (§5.3). These return a *graded* [0,1] (not a hard 0/1) so the
convex gate (gating.py) still produces a smooth firing curve.
"""

from __future__ import annotations

import math

from yaho_pipeline import NegativityScorer
from parapara_pipeline import DepartureScorer

# --------------------------------------------------------------------------- #
# Negativity (§5.3)
# --------------------------------------------------------------------------- #

# Cue -> intensity weight. Substring match (Korean stems vary by ending).
_NEG_CUES: dict[str, float] = {
    "우울": 1.0, "짜증": 1.0, "싫": 0.9, "혼나": 1.0, "혼날": 1.0, "망쳤": 1.0, "망해": 0.9,
    "망하": 0.9, "떨려": 0.8, "무서": 0.9, "창피": 0.9, "쪽팔": 1.0, "욕": 0.7, "스트레스": 0.9,
    "힘들": 0.7, "죽겠": 1.0, "최악": 1.0, "별로": 0.6, "안돼": 0.7, "못해": 0.6, "실패": 0.9,
    "싸웠": 1.0, "싸움": 0.9, "화나": 0.9, "슬프": 0.9, "슬퍼": 0.9, "걱정": 0.7, "불안": 0.8,
    "피곤": 0.5, "지쳤": 0.8, "한심": 0.8, "배신": 0.9, "억울": 0.9, "속상": 0.9, "찍혔": 0.8,
    "잔소리": 0.7, "거지같": 1.0, "쪘": 0.5, "까졌": 0.6, "실수": 0.6, "후회": 0.8, "외로": 0.8,
    "ㅠ": 0.4, "ㅜ": 0.4,
}
_AMPLIFIERS: dict[str, float] = {
    "진짜": 1.3, "너무": 1.3, "완전": 1.2, "개": 1.3, "존나": 1.4, "또": 1.15, "제일": 1.2, "매우": 1.2,
}


class LexiconNegativityScorer(NegativityScorer):
    """Graded negativity in [0,1] from cue density × amplifiers, saturated.

    score = 1 - exp(-k · Σ weights · amp).  k controls how fast strong negativity
    saturates toward 1.
    """

    def __init__(self, k: float = 0.55):
        self.k = k

    def score(self, text: str) -> float:
        base = sum(w for cue, w in _NEG_CUES.items() if cue in text)
        amp = 1.0
        for a, m in _AMPLIFIERS.items():
            if a in text:
                amp *= m
        raw = base * amp
        return 1.0 - math.exp(-self.k * raw)


# --------------------------------------------------------------------------- #
# Departure / closing intent (§5.4)
# --------------------------------------------------------------------------- #

_DEP_CUES: dict[str, float] = {
    "갈게": 1.0, "갈래": 0.9, "가야": 0.9, "가볼게": 1.0, "나갈": 0.9, "나가": 0.7, "이만": 1.0,
    "잘게": 1.0, "잘자": 1.0, "자야": 0.9, "자러": 0.9, "안녕": 0.7, "바이": 0.8, "빠이": 0.8,
    "다녀올": 0.9, "다녀와": 0.6, "들어갈": 0.7, "출근": 0.8, "퇴근": 0.6, "하러": 0.7, "으러": 0.7,
    "씻고": 0.7, "씻을": 0.7, "먼저": 0.5, "또 보자": 0.9, "다음에": 0.6, "끊을": 0.9, "끊자": 0.9,
    "마치고": 0.6, "바빠서": 0.8, "약속": 0.6, "심부름": 0.7, "출발": 0.7,
}


class LexiconDepartureScorer(DepartureScorer):
    """Graded departure/closing intent in [0,1] (§5.4 fallback)."""

    def __init__(self, k: float = 0.7):
        self.k = k

    def score(self, text: str) -> float:
        raw = sum(w for cue, w in _DEP_CUES.items() if cue in text)
        return 1.0 - math.exp(-self.k * raw)
