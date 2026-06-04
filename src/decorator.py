"""§5.2 — Decorator D_rule (the mechanical gyaru decoration layer).

OFFLINE TEACHER, training-time only (§5.6). Takes a first-pass gyaru sentence and
injects interjections / katakana / emojis / kaomoji / tails per the §5.2 rules,
then runs the mandatory postprocessor (rule 6). Only the four mechanical
categories are injected here; context-dependent categories are the stylizer's
job (§5.2 note, see lexicon.MECHANICAL_CATEGORIES / CONTEXT_ONLY_CATEGORIES).

Phase 1a deliverable. Pseudocode (from §5.2):

    decorate(text, λ):
      clauses ← split(text on . ! ? ~)
      for each clause c:
        with prob λ:  prepend a marker (+ TAIL w.p. 0.7; original w.p. P_kata)
        for each word boundary: with prob λ/2 insert an emoji/kaomoji
        c ← c + ENDING + 1..3 emojis ; with prob 0.5 append a kaomoji
      return postprocess(join(clauses))   # rule 6: dedupe runs, 1 ending/clause
"""

from __future__ import annotations

import random


def decorate(text: str, lam: float, *, rng: random.Random | None = None) -> str:
    """Apply §5.2 decoration at density ``lam`` (in [0.2, 0.6]). Phase 1a."""
    raise NotImplementedError("Phase 1a — implement §5.2 rules 1-6 over lexicon.py")


def postprocess(text: str) -> str:
    """§5.2 rule 6 — collapse repeated tails/emojis (~~~~~ -> ~~), 1 ending per
    clause, tidy whitespace. Phase 1a."""
    raise NotImplementedError("Phase 1a — §5.2 rule 6")
