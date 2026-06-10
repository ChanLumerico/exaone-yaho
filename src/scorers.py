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
import re

from yaho_pipeline import NegativityScorer
from parapara_pipeline import DepartureScorer

# --------------------------------------------------------------------------- #
# Negativity (§5.3)
# --------------------------------------------------------------------------- #

# Cue -> intensity weight. Substring match (Korean stems vary by ending).
_NEG_CUES: dict[str, float] = {
    "우울": 1.0, "짜증": 1.0, "싫": 0.9, "혼나": 1.0, "혼날": 1.0, "망쳤": 1.0, "망해": 0.9,
    "망하": 0.9, "망함": 0.9, "떨려": 0.8, "무서": 0.9, "창피": 0.9, "쪽팔": 1.0, "욕먹": 0.8, "욕해": 0.8,
    "스트레스": 0.9,
    "힘들": 0.7, "죽겠": 1.0, "최악": 1.0, "별로": 0.6, "안돼": 0.7, "못해": 0.6, "실패": 0.9,
    "싸웠": 1.0, "싸움": 0.9, "화나": 0.9, "화남": 0.9, "빡쳐": 1.0, "빡치": 1.0, "슬프": 0.9, "슬퍼": 0.9,
    "걱정": 0.7, "불안": 0.8, "피곤": 0.5, "지쳤": 0.8, "지친": 0.8, "한심": 0.8, "배신": 0.9, "억울": 0.9,
    "속상": 0.9, "찍혔": 0.8, "잔소리": 0.7, "거지같": 1.0, "쪘": 0.5, "까졌": 0.6, "실수": 0.6,
    "후회": 0.8, "외로": 0.8, "ㅠ": 0.4, "ㅜ": 0.4,
    # v3 — common Korean negativity the old lexicon missed (yaho never fired on these)
    "현타": 1.0, "현자타임": 1.0, "멘붕": 1.0, "멘탈": 0.7, "노답": 0.9, "답없": 0.9, "답이 없": 0.9,
    "큰일": 0.8, "어떡": 0.8, "어떻해": 0.8, "막막": 0.9, "두려": 0.9, "긴장": 0.7, "초조": 0.8,
    "떨어졌": 0.9, "떨어져": 0.8, "차였": 1.0, "차이": 0.7, "헤어": 0.9, "이별": 0.9, "혼자": 0.5,
    "외톨": 0.9, "소외": 0.9, "상처": 0.8, "아파": 0.6, "아프": 0.6, "눈물": 0.8, "울었": 0.8, "울고": 0.8,
    "그만두": 0.7, "포기": 0.8, "자괴": 0.9, "자존감": 0.7, "쓸모없": 1.0, "한숨": 0.7, "짜증나": 1.0,
    "꿀꿀": 0.7, "우중충": 0.7, "서럽": 0.9, "비참": 1.0, "절망": 1.0, "허무": 0.8, "공허": 0.8,
    "지겨": 0.7, "지긋지긋": 0.9, "토나": 0.8, "괴로": 0.9, "버겁": 0.8, "벅차": 0.6, "주눅": 0.8,
    "눈치": 0.5, "민망": 0.7, "부담": 0.6, "압박": 0.7, "조급": 0.7, "불행": 1.0, "답답": 0.8,
    "깨졌": 0.8, "깨져": 0.7, "털렸": 0.8, "혼쭐": 0.9, "짤렸": 0.9, "짤림": 0.9, "갈굼": 0.8,
    "갈궈": 0.8, "쪼였": 0.7, "데였": 0.6, "구박": 0.8, "타박": 0.7, "엎어": 0.7, "망신": 0.9,
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
    "갈게": 1.0, "갈래": 0.9, "가야": 0.9, "가볼게": 1.0, "나갈": 0.9, "나가서": 0.7, "나갔다": 0.8,
    "이만": 1.0, "잘게": 1.0, "잘자": 1.0, "잘 자": 1.0, "자러": 0.9, "바이": 0.8, "빠이": 0.8,
    "다녀올": 0.9, "다녀와": 0.6, "다녀오": 0.7, "들어갈": 0.7, "들어가": 0.6, "출근": 0.8, "퇴근": 0.6,
    "하러": 0.7, "으러": 0.7, "씻고": 0.7, "씻을": 0.7, "씻으러": 0.8, "먼저": 0.5, "또 보자": 0.9,
    "또 봐": 0.8, "다음에": 0.5, "끊을": 0.9, "끊자": 0.9, "끊어": 0.7, "마치고": 0.6, "바빠서": 0.7,
    "약속": 0.5, "심부름": 0.7, "출발": 0.7, "나가봐야": 0.9, "가봐야": 0.9, "올게": 0.7, "와야": 0.6,
    # NOTE: bare "안녕" REMOVED (v3) — it is overwhelmingly a GREETING in the seed,
    # which made parapara misfire on self-introductions. Real farewells are covered
    # by the explicit departure verbs above + the greeting guard below.
}

# v3 — greeting / self-introduction guard. If the turn is a greeting or intro and
# has no strong departure verb, it is NOT a departure (kills the "안녕, 나는 …" misfire).
_GREETING_RE = re.compile(r"(안녕|반가|처음 ?뵙|소개|^하이|^헬로|나는?\s*\d{0,2}대|이야기?\s*하자|만나서)")
_STRONG_DEP = ("갈게", "가야", "가볼", "나갈", "나가봐", "가봐야", "이만", "잘게", "자러", "다녀",
               "끊", "출발", "올게", "들어갈", "퇴근", "출근")


class LexiconDepartureScorer(DepartureScorer):
    """Graded departure/closing intent in [0,1] (§5.4 fallback)."""

    def __init__(self, k: float = 0.7):
        self.k = k

    def score(self, text: str) -> float:
        # greeting/intro with no strong departure verb -> not leaving (§5.5 misfire fix)
        if _GREETING_RE.search(text) and not any(d in text for d in _STRONG_DEP):
            return 0.0
        raw = sum(w for cue, w in _DEP_CUES.items() if cue in text)
        return 1.0 - math.exp(-self.k * raw)
