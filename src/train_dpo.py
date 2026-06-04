"""§7.6 / §7.4(d) — Stage-2 preference optimization (Phase 2.5, POST-MVP / optional).

mlx-lm has no built-in DPO, so the path splits (§7.6 / §16.5):
  * ORPO (preferred, ref-free -> memory/speed win), via a community/MLX impl, or
  * PyTorch-MPS + TRL DPO (slower, op fallbacks).
Online RL (PPO/GRPO) is last-resort and wants a short cloud-GPU rental.

L_DPO = -E[ log σ( β·log(π_θ(y+)/π_ref(y+)) - β·log(π_θ(y-)/π_ref(y-)) ) ]
β = style-strength <-> stability dial (sweep 0.1/0.2/0.3, §16.3).

SFT alone already learns both memes + persona (§7.6) — this only raises quality.
"""

from __future__ import annotations


def run(config_path: str = "configs/dpo.yaml") -> None:
    """ORPO/DPO from the SFT adapter + JSONL reward logging (§17.2). Phase 2.5."""
    raise NotImplementedError("Phase 2.5 (post-MVP, optional) — §7.6, prefer ORPO")
