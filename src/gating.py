"""§5.3 / §5.4 / §5.5 — convex gating math + two-policy arbitration.

This is pure, unambiguous math (no model dependencies), so it is implemented and
unit-tested from the start. The *scorers* that produce the [0,1] inputs, the
noun/activity *extractors*, and the *renderers* remain Phase-1 work (stubs in
``yaho_pipeline.py`` / ``parapara_pipeline.py``).

Gate (convex, gamma > 1 -> suppress weak signal, fire hard on strong signal):

    p = p_base + (p_max - p_base) * s ** gamma        with s in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class GateParams:
    """Parameters for one convex gate (loaded from configs/gate.yaml)."""

    p_base: float
    p_max: float
    gamma: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_base <= self.p_max <= 1.0):
            raise ValueError("require 0 <= p_base <= p_max <= 1")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")


def gate_prob(score: float, params: GateParams) -> float:
    """Map a signal strength ``score`` in [0,1] to a firing probability.

    Convex in ``score`` for gamma > 1. Clamps ``score`` to [0,1].
    """
    s = 0.0 if score < 0.0 else 1.0 if score > 1.0 else score
    return params.p_base + (params.p_max - params.p_base) * (s ** params.gamma)


class Policy(str, Enum):
    """Which behavioral policy wins arbitration for a turn (§5.5)."""

    YAHO = "yaho"
    PARAPARA = "parapara"
    PLAIN = "plain"  # neither fires -> ordinary gyaru decoration


def arbitrate(
    *,
    s_neg: float,
    s_dep: float,
    yaho_params: GateParams,
    parapara_params: GateParams,
    departure_priority_threshold: float,
    rng_yaho: float,
    rng_para: float,
) -> Policy:
    """§5.5 arbitration between yaho (negativity) and parapara (departure).

    Rules:
      1. Compute both firing probabilities.
      2. If the departure signal is strong (``s_dep`` >= threshold), parapara
         wins when it fires (farewell reads naturally as parapara).
      3. Otherwise the higher firing probability wins.
      4. If neither clears its Bernoulli draw, fall back to PLAIN.
      5. Never both in one response (caller emits a single policy).

    ``rng_yaho`` / ``rng_para`` are the caller's uniform[0,1) draws (injected for
    determinism/testability). A policy "fires" iff its draw < its probability.
    """
    p_yaho = gate_prob(s_neg, yaho_params)
    p_para = gate_prob(s_dep, parapara_params)

    yaho_fires = rng_yaho < p_yaho
    para_fires = rng_para < p_para

    # Rule 2 — strong departure prefers parapara.
    if s_dep >= departure_priority_threshold and para_fires:
        return Policy.PARAPARA

    if yaho_fires and para_fires:
        # Rule 3 — higher probability wins (tie -> parapara, farewell-natural).
        return Policy.YAHO if p_yaho > p_para else Policy.PARAPARA
    if yaho_fires:
        return Policy.YAHO
    if para_fires:
        return Policy.PARAPARA
    return Policy.PLAIN
