"""§5.3 / §5.4 — concrete renderers for the two meme policies.

These render the signature meme phrases + light §5.2-style decoration (emoji /
kaomoji / tail). They deliberately do NOT run the full mechanical decorator: the
deflection / parapara line is short and its core phrase is the meme signature, so
only surface decoration is added (the noun/"야호~" and "파라파라나 추고있어야겠다~"
stay intact). ``rng`` injected for determinism.
"""

from __future__ import annotations

import random

import lexicon as lx
from yaho_pipeline import Renderer as _YahoRendererBase
from parapara_pipeline import Renderer as _ParaRendererBase


# v3 — diversify: sample evenly across the WIDE languid emoji bank (no 🙄
# monoculture), but never repeat the same emoji twice in one render.
def _emojis(rng: random.Random, n_range=(1, 2)) -> str:
    n = rng.randint(*n_range)
    pool = list(lx.EMOJI_BANK)
    rng.shuffle(pool)
    return "".join(pool[:n])


class YahoRenderer(_YahoRendererBase):
    """noun -> "{noun} 야호~" + 1–2 emojis (+ optional tail / kaomoji)."""

    def __init__(self, rng: random.Random | None = None, *, p_extra_tail=0.3, p_kaomoji=0.3):
        self.rng = rng or random.Random()
        self.p_extra_tail = p_extra_tail
        self.p_kaomoji = p_kaomoji

    def render(self, noun: str) -> str:
        out = lx.YAHO_TEMPLATE.format(noun=noun)          # "{noun} 야호~"
        if self.rng.random() < self.p_extra_tail:
            out += self.rng.choice(["~", "~~"])
        out += _emojis(self.rng)
        if self.rng.random() < self.p_kaomoji:
            out += " " + self.rng.choice(lx.KAOMOJI)
        return out


class ParaParaRenderer(_ParaRendererBase):
    """activity|None -> the §5.4 line + 💖 signature (+ optional emoji/kaomoji)."""

    def __init__(self, rng: random.Random | None = None, *, p_extra_emoji=0.5, p_kaomoji=0.3):
        self.rng = rng or random.Random()
        self.p_extra_emoji = p_extra_emoji
        self.p_kaomoji = p_kaomoji

    def render(self, activity: str | None) -> str:
        # v3 — frame VARIETY + safe fallback. Use a grammatical "{activity}" frame
        # only ~45% of the time AND only when a clean gerund was extracted; else an
        # activity-free frame (always grammatical for any departure). Prevents the
        # rigid "약속하는 동안" slot-fill the single old frame caused.
        if activity and self.rng.random() < 0.45:
            out = self.rng.choice(lx.PARAPARA_ACTIVITY_FRAMES).format(activity=activity)
        else:
            out = self.rng.choice(lx.PARAPARA_GENERIC_FRAMES)
        out += lx.PARAPARA_SIGNATURE_EMOJI                 # 💖 signature
        if self.rng.random() < 0.75:                       # v4 — 맛데루용/오이데 시너지 (확률↑)
            out += " " + self.rng.choice(lx.PARAPARA_COMPANION)
        if self.rng.random() < self.p_extra_emoji:
            out += _emojis(self.rng, (1, 1))
        if self.rng.random() < self.p_kaomoji:
            out += " " + self.rng.choice(lx.KAOMOJI)
        return out
