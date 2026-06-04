"""§6 / §7.4(d) — Preference-pair synthesis for Stage-2 (Phase 1e, POST-MVP).

For each `chosen` (right density, context-appropriate, yaho only when correct,
diverse markers, meaning preserved) generate MULTIPLE `rejected` types so the
alignment signal is rich (§7.4d):
  1. under_styling   (밋밋함)
  2. over_styling    (도배 / emoji spam, content lost)
  3. yaho_misfire    (야호 in a neutral context)
  4. semantic_drift  (off-context / repetitive)

Generation: sample N from the SFT model, label pairs by LLM-judge or rule
(styleScore / sim / yaho-appropriateness). Emit data/prefs.jsonl.
"""

from __future__ import annotations

from pathlib import Path


def build_preference_pairs(sft_model: str, config_path: str = "configs/dpo.yaml") -> Path:
    """Emit data/prefs.jsonl (chosen + 4 rejected types). Phase 1e."""
    raise NotImplementedError("Phase 1e (post-MVP) — §7.4d")
