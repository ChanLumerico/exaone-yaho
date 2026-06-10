"""Assemble the yaho / parapara pipelines from configs/gate.yaml (§11).

Keeps the drop-in design (§4): the scorer backend is chosen by config
(`scorer: lexicon|koelectra|sent5|nli`). Phase 1b/1c ship the lexicon backends;
dense backends raise NotImplementedError until added.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from arbiter import MemeArbiter
from gating import GateParams
from extractors import KiwiNounExtractor, RuleActivityExtractor
from parapara_pipeline import ParaParaPipeline
from renderers import ParaParaRenderer, YahoRenderer
from scorers import LexiconDepartureScorer, LexiconNegativityScorer
from yaho_pipeline import YahoPipeline

_ROOT = Path(__file__).resolve().parent.parent


def _load(config_path: str | Path) -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = _ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def build_yaho_pipeline(config_path="configs/gate.yaml", *, rng=None, kiwi=None) -> YahoPipeline:
    cfg = _load(config_path)
    y, ne = cfg["yaho"], cfg["noun_extractor"]
    params = GateParams(p_base=y["p_base"], p_max=y["p_max"], gamma=y["gamma"])
    if y["scorer"] == "lexicon":
        scorer = LexiconNegativityScorer()
    else:  # koelectra | sent5 — dense backends, Phase 1b+
        raise NotImplementedError(f"NegativityScorer backend {y['scorer']!r} not yet implemented")
    extractor = KiwiNounExtractor(exclude_time=tuple(ne["exclude_time_nouns"]), kiwi=kiwi)
    return YahoPipeline(scorer, extractor, YahoRenderer(rng=rng or random.Random()), params)


def build_parapara_pipeline(config_path="configs/gate.yaml", *, rng=None, kiwi=None) -> ParaParaPipeline:
    cfg = _load(config_path)
    p = cfg["parapara"]
    params = GateParams(p_base=p["p_base"], p_max=p["p_max"], gamma=p["gamma"])
    if p["scorer"] == "lexicon":
        scorer = LexiconDepartureScorer()
    else:  # nli — dense backend, Phase 1c+
        raise NotImplementedError(f"DepartureScorer backend {p['scorer']!r} not yet implemented")
    extractor = RuleActivityExtractor(kiwi=kiwi)
    return ParaParaPipeline(scorer, extractor, ParaParaRenderer(rng=rng or random.Random()), params)


def build_arbiter(config_path="configs/gate.yaml", *, rng=None, kiwi=None) -> MemeArbiter:
    """§5.5 — build a MemeArbiter wiring both pipelines (shares one Kiwi)."""
    rng = rng or random.Random()
    if kiwi is None:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
    arb = _load(config_path)["arbitration"]
    return MemeArbiter(
        yaho=build_yaho_pipeline(config_path, rng=rng, kiwi=kiwi),
        parapara=build_parapara_pipeline(config_path, rng=rng, kiwi=kiwi),
        departure_priority_threshold=arb["departure_priority_threshold"],
    )
