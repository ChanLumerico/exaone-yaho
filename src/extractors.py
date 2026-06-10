"""§5.3 / §5.4 / §16.6 — concrete extractors (noun / activity).

KiwiNounExtractor pulls the salient noun for the "{noun} 야호~" deflection.
Saliency priority (§5.3/§16.6): proper noun (NNP) > object common noun (NNG + 을/를)
> first general noun (frequency proxy). Time nouns excluded; None when nothing
usable (-> firing is skipped, avoids awkward output).

The Kiwi model is heavy to construct, so a single instance can be shared across
extractors via the ``kiwi=`` argument.
"""

from __future__ import annotations

import re

from yaho_pipeline import NounExtractor
from parapara_pipeline import ActivityExtractor

_DEFAULT_TIME_NOUNS = ("오늘", "내일", "어제", "지금", "요즘", "이번", "다음", "아까", "방금")


class KiwiNounExtractor(NounExtractor):
    def __init__(self, exclude_time=_DEFAULT_TIME_NOUNS, kiwi=None):
        from kiwipiepy import Kiwi

        self.kiwi = kiwi or Kiwi()
        self.exclude = set(exclude_time)

    def extract(self, text: str) -> str | None:
        toks = self.kiwi.tokenize(text)
        proper, obj, general = [], [], []
        for i, t in enumerate(toks):
            if t.tag not in ("NNP", "NNG") or t.form in self.exclude:
                continue
            is_object = i + 1 < len(toks) and toks[i + 1].tag == "JKO"  # 을/를
            if t.tag == "NNP":
                proper.append(t.form)
            elif is_object:
                obj.append(t.form)
            else:
                general.append(t.form)
        for bucket in (proper, obj, general):  # §5.3 priority order
            if bucket:
                return bucket[0]
        return None  # §16.6 — no usable noun -> skip firing


class RuleActivityExtractor(ActivityExtractor):
    """§5.4 — extract a GRAMMATICAL gerund ("출근하는", "자는") or None (-> generic).

    v3: the renderer slots the result into "네가 {activity} 동안 …", so {activity}
    must already be a well-formed "~는" gerund. We only return one when CONFIDENT
    (known activity keyword, or a clean "X하러/X하고" → "X하는"); otherwise None, and
    the renderer falls back to an always-grammatical activity-free frame. This kills
    the old "약속" → "약속하는 동안" ungrammatical slot-fill.
    """

    # explicit noun/stem -> grammatical gerund. Covers the common departures.
    _GERUND = {
        "출근": "출근하는", "퇴근": "퇴근하는", "공부": "공부하는", "운동": "운동하는",
        "샤워": "샤워하는", "알바": "알바하는", "아르바이트": "알바하는", "수업": "수업 듣는",
        "회의": "회의하는", "산책": "산책하는", "청소": "청소하는", "요리": "요리하는",
        "게임": "게임하는", "일": "일하는", "장": "장 보는", "장보": "장 보는",
        "쇼핑": "쇼핑하는", "데이트": "데이트하는", "여행": "여행하는", "출장": "출장 가있는",
        "병원": "병원 가는", "학교": "학교 가는", "회사": "회사 가는", "면접": "면접 보는",
        "씻": "씻는", "자": "자는", "먹": "밥 먹는", "밥": "밥 먹는",
    }
    _GOING = re.compile(r"(가|간다|갈|가야|가자|갔|나가|나갔|다녀|들어가|출발|이만|먼저|올게|와야)")
    # "X하러 …", "X하고 …", "X 하러" → stem X
    _PURPOSE = re.compile(r"([가-힣]{1,6}?)하(?:러|고)(?=[\s가-힣])")  # v4 — require 하 (만나러 X)

    def __init__(self, kiwi=None):
        self.kiwi = kiwi  # optional; the rule covers the common cases

    def extract(self, text: str) -> str | None:
        if not self._GOING.search(text):
            return None
        # 1) direct keyword hit -> known grammatical gerund (longest key first)
        for key in sorted(self._GERUND, key=len, reverse=True):
            if key in text:
                return self._GERUND[key]
        # 2) "X하러/X하고" purpose -> "X하는" only if X is a clean 하다-noun (>=2 chars)
        m = self._PURPOSE.search(text)
        if m and m.group(1) and len(m.group(1)) >= 2:
            return m.group(1) + "하는"
        return None  # uncertain -> generic activity-free frame (always grammatical)
