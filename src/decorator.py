"""§5.2 — Decorator D_rule (the mechanical gyaru decoration layer).

OFFLINE TEACHER, training-time only (§5.6). Takes a first-pass gyaru sentence
(from the stylizer S_LLM) and injects interjections / katakana / emoji / kaomoji
/ tails per the §5.2 rules, then runs the mandatory postprocessor (rule 6).

Only the four MECHANICAL categories are injected here; context-dependent
categories (affection/fandom/greeting/reaction) are the stylizer's job
(§5.2 note, see lexicon.MECHANICAL_CATEGORIES / CONTEXT_ONLY_CATEGORIES).

All probabilities come from configs/style.yaml (§11). ``rng`` is injected so the
decoration is deterministic/reproducible under a fixed seed (§11/§17).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import lexicon as lx

_ROOT = Path(__file__).resolve().parent.parent
_TERMINATORS = ".!?~"


@dataclass(frozen=True)
class DecoratorParams:
    """§5.2 knobs, loaded from configs/style.yaml (no hardcoding, §11)."""

    lambda_range: tuple[float, float]
    p_kata: float
    p_tail_after_marker: float
    emoji_boundary_prob_factor: float
    ending_emoji_count: tuple[int, int]
    p_trailing_kaomoji: float
    max_ending_per_clause: int
    collapse_repeats: bool
    max_tail_run: int
    mechanical_categories: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/style.yaml") -> "DecoratorParams":
        import yaml

        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {})["decorator"]
        pp = d["postprocess"]
        return cls(
            lambda_range=tuple(d["lambda_range"]),
            p_kata=float(d["p_kata"]),
            p_tail_after_marker=float(d["p_tail_after_marker"]),
            emoji_boundary_prob_factor=float(d["emoji_boundary_prob_factor"]),
            ending_emoji_count=tuple(d["ending_emoji_count"]),
            p_trailing_kaomoji=float(d["p_trailing_kaomoji"]),
            max_ending_per_clause=int(pp["max_ending_per_clause"]),
            collapse_repeats=bool(pp["collapse_repeats"]),
            max_tail_run=int(pp["max_tail_run"]),
            mechanical_categories=tuple(d["mechanical_categories"]),
        )


def _split_clauses(text: str) -> list[str]:
    """Split on sentence punctuation . ! ? ~ keeping the terminators (§5.2)."""
    return [c for c in re.findall(r"[^.!?~]*[.!?~]+|[^.!?~]+", text) if c.strip()]


# §5.2 rule 2 — katakana/kanji pass (for ALREADY-styled text from S_LLM). The yaho
# meme is excluded so it stays Korean (파라파라 isn't in the lexicon anyway).
_KATA_PAIRS = sorted(
    [(t, o) for e in lx.LEXICON.values() for (t, o) in e if "야호" not in t],
    key=lambda p: -len(p[0]),
)
_LETTER_RE = re.compile(r"[가-힣A-Za-z]")


def katakana_substitute(text: str, p_kata: float, rng: random.Random) -> str:
    """Replace each lexicon marker occurrence with its original form w.p. ``p_kata``,
    only at a right word boundary (avoids mangling words like 카와이한)."""
    for translit, original in _KATA_PAIRS:
        i, res = 0, []
        while True:
            j = text.find(translit, i)
            if j == -1:
                res.append(text[i:]); break
            res.append(text[i:j])
            end = j + len(translit)
            nxt = text[end] if end < len(text) else ""
            if ((not nxt) or (not _LETTER_RE.match(nxt))) and rng.random() < p_kata:
                res.append(original)
            else:
                res.append(translit)
            i = end
        text = "".join(res)
    return text


def _decorate_clause(clause: str, lam: float, p: DecoratorParams, rng: random.Random) -> str:
    body = clause.strip()
    if not body:
        return ""

    # rule 1 — interjection prepend (mechanical categories only), w.p. lambda.
    if rng.random() < lam:
        cat = rng.choice(p.mechanical_categories)
        translit, original = rng.choice(lx.LEXICON[cat])
        w = original if rng.random() < p.p_kata else translit          # rule 2 — katakana
        if rng.random() < p.p_tail_after_marker:                        # rule 5 — tail
            w = w + rng.choice(lx.TAILS)
        body = w + " " + body

    # rule 3 — emoji/kaomoji at word boundaries (not just the end).
    words = body.split(" ")
    out = words[0]
    boundary_p = lam * p.emoji_boundary_prob_factor
    for word in words[1:]:
        if rng.random() < boundary_p:
            tok = rng.choice(lx.EMOJI_BANK + lx.KAOMOJI)
            out += (tok + " " if tok in lx.EMOJI_BANK else " " + tok + " ") + word
        else:
            out += " " + word
    body = out

    # rule 5 — hi-tension clause ending: replace the terminator with a TAIL
    # (preserve a question mark), then append 1..3 emojis (rule).
    m = re.search(r"[.!?~]+\s*$", body)
    was_q = "?" in (m.group() if m else "")
    core = (body[: m.start()] if m else body).rstrip()
    body = core + ("?" if was_q else "") + rng.choice(lx.TAILS)
    body += "".join(rng.choice(lx.EMOJI_BANK) for _ in range(rng.randint(*p.ending_emoji_count)))

    # rule 4 — trailing kaomoji, w.p. p_trailing_kaomoji.
    if rng.random() < p.p_trailing_kaomoji:
        body += " " + rng.choice(lx.KAOMOJI)
    return body


def decorate(text: str, lam: float, *, params: DecoratorParams, rng: random.Random) -> str:
    """Apply §5.2 decoration at density ``lam`` (in [0.2, 0.6]), then postprocess."""
    clauses = _split_clauses(text)
    decorated = " ".join(_decorate_clause(c, lam, params, rng) for c in clauses if c.strip())
    return postprocess(decorated, params=params)


def postprocess(text: str, *, params: DecoratorParams) -> str:
    """§5.2 rule 6 — collapse repeated tails/emojis, tidy whitespace."""
    n = params.max_tail_run
    if params.collapse_repeats:
        # collapse runs of tail chars (~ ! ^), mixed or single, to max_tail_run
        text = re.sub(r"[~!^]{" + str(n + 1) + r",}", lambda m: m.group()[:n], text)
        # collapse consecutive duplicate emojis (EMOJI_BANK only; kaomoji left alone)
        for e in lx.EMOJI_BANK:
            text = re.sub("(?:" + re.escape(e) + r"){2,}", e, text)
    text = re.sub(r"[ \t]{2,}", " ", text)            # collapse whitespace
    text = re.sub(r"\s+([.!?~,])", r"\1", text)        # no space before punctuation
    return text.strip()


class Decorator:
    """Stateful convenience: holds params + rng, samples lambda, decorates."""

    def __init__(self, params: DecoratorParams | None = None, *, seed: int | None = None,
                 rng: random.Random | None = None):
        self.params = params or DecoratorParams.from_yaml()
        self.rng = rng or random.Random(seed)

    def sample_lambda(self) -> float:
        lo, hi = self.params.lambda_range
        return self.rng.uniform(lo, hi)

    def __call__(self, text: str, lam: float | None = None) -> str:
        if lam is None:
            lam = self.sample_lambda()
        return decorate(text, lam, params=self.params, rng=self.rng)

    def light(self, text: str) -> str:
        """For ALREADY-styled S_LLM output: §5.2 rule 2 (katakana) + rule 6 only —
        no interjection injection / ending-emoji spam (avoids over-styling rich text)."""
        return postprocess(katakana_substitute(text, self.params.p_kata, self.rng), params=self.params)
