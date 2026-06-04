"""§5.3 — Yaho subsystem (first-class). OFFLINE TEACHER, training-time only (§5.6).

Deliberately ignores context on some turns and deflects with "{핵심명사} 야호~".
The stronger the negativity/pressure of the (previous user) turn, the more often
it fires. Origin meme: "거제 시민들한테 혼나" -> "거제 야호~" (§A.1).

Four interchangeable components (§4 — each an interface for drop-in swap):
  * NegativityScorer : text -> s_neg in [0,1]  (dense, NOT a sparse lexicon branch)
  * NounExtractor    : text -> salient noun | None
  * Gate             : convex map s_neg -> firing prob (see gating.py)
  * Renderer         : "{noun} 야호~" + decoration (§5.2)

Implementations are Phase 1b. This module defines the contracts + the firing
procedure only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from gating import GateParams, gate_prob


class NegativityScorer(ABC):
    """text -> negativity strength in [0,1]. Dense by design (§5.3).

    Drop-in backends (§5.3): (a) dense KoELECTRA sentiment, (b) 5-emotion model
    {anger+fear+sadness} sum, (c) cue lexicon (CI/fallback only).
    """

    @abstractmethod
    def score(self, text: str) -> float:
        raise NotImplementedError


class NounExtractor(ABC):
    """text -> salient noun for the deflection, or None to skip (§5.3 / §16.6).

    Priority: proper noun > object-ish common noun > frequency. Excludes time
    nouns (오늘/내일/어제). Returns None when nothing usable -> firing is skipped.
    """

    @abstractmethod
    def extract(self, text: str) -> str | None:
        raise NotImplementedError


class Renderer(ABC):
    """noun -> rendered "{noun} 야호~" with §5.2 decoration."""

    @abstractmethod
    def render(self, noun: str) -> str:
        raise NotImplementedError


@dataclass
class YahoPipeline:
    """Composes the four components; ``maybe_yaho`` is the §5.3 procedure."""

    scorer: NegativityScorer
    extractor: NounExtractor
    renderer: Renderer
    params: GateParams

    def firing_prob(self, turn: str) -> float:
        return gate_prob(self.scorer.score(turn), self.params)

    def maybe_yaho(self, turn: str, rng: float) -> str | None:
        """Return a rendered yaho deflection, or None (fall back to plain gyaru).

        ``rng`` is a uniform[0,1) draw injected for determinism (§17 reproducibility).
        """
        if rng < self.firing_prob(turn):
            noun = self.extractor.extract(turn)
            if noun is not None:
                return self.renderer.render(noun)
        return None
