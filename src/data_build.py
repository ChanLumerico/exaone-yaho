"""§6 — Multi-turn gyaru corpus synthesis (Phase 1d). OFFLINE (§5.6).

Pipeline (§6 합성 절차):
  1. collect multi-turn seeds (SmileStyle anchor; AI Hub / 모두의 말뭉치 for volume)
  2. S_LLM stylize assistant turns (meme-free gyaru, §16.4)
  3. D_rule decorate (§5.2)
  4. yaho/parapara branch on the *previous user turn* (§5.3/§5.4) + arbitrate (§5.5)
  5. quality filter: sim(x,y) >= tau_sem (~0.55) AND styleScore >= tau_sty + turn consistency
  6. diversity check: marker/emoji distribution monitor, n-gram Jaccard near-dup removal
  7. emit data/sft_multiturn.jsonl (500-1500 dialogues) + mix capability-replay (§7.4b)

Every decision is logged to JSONL (§17.2 phase=data): s_neg, s_dep, p_yaho, p_para,
yaho_fired, parapara_fired, sim, style_score, accepted, reject_reason, ...
"""

from __future__ import annotations

from pathlib import Path


def build_sft_corpus(config_path: str = "configs/style.yaml") -> Path:
    """Run the §6 synthesis pipeline and write data/sft_multiturn.jsonl. Phase 1d."""
    raise NotImplementedError("Phase 1d — see §6 + §16.4; log every sample (§17)")
