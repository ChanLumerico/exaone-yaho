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


def _emojis(rng: random.Random, n_range=(1, 2)) -> str:
    return "".join(rng.choice(lx.EMOJI_BANK) for _ in range(rng.randint(*n_range)))


class YahoRenderer(_YahoRendererBase):
    """noun -> "{noun} 야호~" + 1–2 emojis (+ optional tail / kaomoji)."""

    def __init__(self, rng: random.Random | None = None, *, p_extra_tail=0.3, p_kaomoji=0.4):
        self.rng = rng or random.Random()
        self.p_extra_tail = p_extra_tail
        self.p_kaomoji = p_kaomoji

    def render(self, noun: str) -> str:
        out = lx.YAHO_TEMPLATE.format(noun=noun)          # "{noun} 야호~"
        if self.rng.random() < self.p_extra_tail:
            out += self.rng.choice(["~", "!", "^"])
        out += _emojis(self.rng)
        if self.rng.random() < self.p_kaomoji:
            out += " " + self.rng.choice(lx.KAOMOJI)
        return out


class ParaParaRenderer(_ParaRendererBase):
    """activity|None -> the §5.4 line + 💖 signature (+ optional emoji/kaomoji)."""

    def __init__(self, rng: random.Random | None = None, *, p_extra_emoji=0.5, p_kaomoji=0.4):
        self.rng = rng or random.Random()
        self.p_extra_emoji = p_extra_emoji
        self.p_kaomoji = p_kaomoji

    def render(self, activity: str | None) -> str:
        if activity:
            out = lx.PARAPARA_TEMPLATE.format(activity=activity)
        else:
            out = lx.PARAPARA_GENERIC
        out += lx.PARAPARA_SIGNATURE_EMOJI                 # 💖 signature
        if self.rng.random() < self.p_extra_emoji:
            out += _emojis(self.rng, (1, 1))
        if self.rng.random() < self.p_kaomoji:
            out += " " + self.rng.choice(lx.KAOMOJI)
        return out
