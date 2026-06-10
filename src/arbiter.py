"""§5.5 — arbitration between the two meme policies for a single turn.

Wires the yaho + parapara pipelines through gating.arbitrate so EXACTLY ONE
policy is emitted per turn (never both, §5.5). Used by Phase 1d synthesis to
decide what (if anything) replaces / augments the stylized assistant turn.

Decision (§5.5): strong departure -> parapara; else the higher firing prob; if
neither clears its draw -> PLAIN (fall back to ordinary gyaru decoration). If
yaho is chosen but no usable noun exists, firing is skipped (§16.6) -> PLAIN.
"""

from __future__ import annotations

from dataclasses import dataclass

from gating import Policy, arbitrate
from parapara_pipeline import ParaParaPipeline
from yaho_pipeline import YahoPipeline


@dataclass
class MemeArbiter:
    yaho: YahoPipeline
    parapara: ParaParaPipeline
    departure_priority_threshold: float

    def decide(self, turn: str, *, rng_yaho: float, rng_para: float) -> tuple[Policy, str | None]:
        """Return (policy, rendered_text|None) for the (previous-user) ``turn``.

        ``rng_yaho``/``rng_para`` are uniform[0,1) draws (injected for determinism).
        """
        s_neg = self.yaho.scorer.score(turn)
        s_dep = self.parapara.scorer.score(turn)
        policy = arbitrate(
            s_neg=s_neg, s_dep=s_dep,
            yaho_params=self.yaho.params, parapara_params=self.parapara.params,
            departure_priority_threshold=self.departure_priority_threshold,
            rng_yaho=rng_yaho, rng_para=rng_para,
        )
        if policy is Policy.YAHO:
            noun = self.yaho.extractor.extract(turn)
            if noun is None:                       # §16.6 — no usable noun -> skip
                return Policy.PLAIN, None
            return Policy.YAHO, self.yaho.renderer.render(noun)
        if policy is Policy.PARAPARA:
            activity = self.parapara.extractor.extract(turn)  # None -> generic
            return Policy.PARAPARA, self.parapara.renderer.render(activity)
        return Policy.PLAIN, None
