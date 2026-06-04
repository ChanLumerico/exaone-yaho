"""§8 / §16 — Evaluation (Phase 3). Produces reproducible reports + JSONL (§17).

Metrics (§8):
  * Acc_style      — gyaru vs neutral classifier (NOT the data-filter styleScore, §16.2)
  * Sim            — meaning preservation (sentence-embedding cosine; back-translation
                     / QA-consistency as auxiliary, BERTScore alone is insufficient, §16.2)
  * multiturn_consistency — fraction of turns holding gyaru markers + persona-drift count
  * yaho/parapara curves  — firing rate per negativity/departure bin vs the §5.3/§5.4
                     target curves (report MAE); event=curve in JSONL (§17.2)
  * win-rate       — SFT vs DPO via LLM-judge (rubric fixed, position-bias controlled)
  * ppl, J = Acc_style * Sim

Capability retention (§16.1): KoBEST/KMMLU subset or held-out instruction-following
50 Q, before vs after; event=capability. Ablation matrix (§16.3): SFT vs +DPO,
dense-gate vs lexicon-gate, replay 0% vs 10-20%, β sweep.

Fixed eval set (gold-based) + fixed seed -> reports/.
"""

from __future__ import annotations

from pathlib import Path


def run_eval(model: str, eval_set: str = "data/gold", ablation_id: str = "base") -> Path:
    """Evaluate ``model`` on the fixed eval set, log to JSONL, write a report. Phase 3."""
    raise NotImplementedError("Phase 3 — §8 + §16; log every example + aggregate (§17)")
