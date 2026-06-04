"""§5.4 — ParaPara subsystem (second-class, symmetric to yaho). OFFLINE TEACHER (§5.6).

When the user ends the conversation or leaves / goes to do something, deflect
with "네가 {활동}하는 동안, 그럼 난 파라파라나 추고있어야겠다~💖". If no activity
can be extracted, use the generic template. Lore: §A.5 (Eurobeat gyaru club
dance -> a behavioral policy that fills idle / farewell / waiting contexts).

Four interchangeable components (§4), parallel to yaho:
  * DepartureScorer   : text -> s_dep in [0,1]  (dense = zero-shot NLI entailment)
  * ActivityExtractor : text -> activity phrase | None
  * Gate              : convex map (see gating.py)
  * Renderer          : template fill + decoration (💖 signature)

Implementations are Phase 1c. Contracts + firing procedure only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from gating import GateParams, gate_prob


class DepartureScorer(ABC):
    """text -> departure/closing intent strength in [0,1] (§5.4).

    Dense backend: zero-shot NLI entailment of the hypothesis "the speaker is
    ending the conversation or leaving". Cue lexicon = fallback.
    """

    @abstractmethod
    def score(self, text: str) -> float:
        raise NotImplementedError


class ActivityExtractor(ABC):
    """text -> the activity the user is leaving to do, or None (§5.4 / §16.6).

    Looks for "~하러/~으러 가" patterns or a verb phrase. None -> generic template.
    """

    @abstractmethod
    def extract(self, text: str) -> str | None:
        raise NotImplementedError


class Renderer(ABC):
    """activity|None -> rendered parapara line with §5.2 decoration (💖)."""

    @abstractmethod
    def render(self, activity: str | None) -> str:
        raise NotImplementedError


@dataclass
class ParaParaPipeline:
    """Composes the four components; ``maybe_parapara`` is the §5.4 procedure."""

    scorer: DepartureScorer
    extractor: ActivityExtractor
    renderer: Renderer
    params: GateParams

    def firing_prob(self, turn: str) -> float:
        return gate_prob(self.scorer.score(turn), self.params)

    def maybe_parapara(self, turn: str, rng: float) -> str | None:
        """Return a rendered parapara line, or None (fall back to plain gyaru).

        Unlike yaho, a None activity does NOT skip firing — it renders the
        generic template (§5.4). ``rng`` is an injected uniform[0,1) draw.
        """
        if rng < self.firing_prob(turn):
            activity = self.extractor.extract(turn)  # may be None -> generic
            return self.renderer.render(activity)
        return None
