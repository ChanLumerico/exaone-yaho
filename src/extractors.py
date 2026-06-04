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
    """§5.4 — extract the activity the user leaves to do, or None (-> generic).

    Two-step: require a "going" verb somewhere in the turn (가/간다/갈/...), then
    pull the purpose stem before "~하러/~으러/~러" (e.g. "공부하러 도서관 간다" ->
    "공부", "밥 먹으러 가" -> "먹"). Words may sit between the purpose and the verb.
    """

    _PURPOSE = re.compile(r"([가-힣]{1,8}?)(?:하|으)?러(?=[\s가-힣])")
    _GOING = re.compile(r"(가|간다|갈|가야|가자|갔|나가|다녀|출발)")

    def __init__(self, kiwi=None):
        self.kiwi = kiwi  # optional; the rule covers the common cases

    def extract(self, text: str) -> str | None:
        if not self._GOING.search(text):
            return None
        m = self._PURPOSE.search(text)
        return m.group(1) if m and m.group(1) else None
