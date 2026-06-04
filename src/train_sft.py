"""§7.3 — Stage-1 SFT via mlx-lm LoRA (Phase 2).

The Phase-0 smoke test validated this exact path (convert -> mlx_lm.lora -> adapter,
loss decreasing). mlx_lm.lora loads with trust_remote_code=True internally, so no
extra flag is needed for EXAONE-3.5 under the pinned toolchain (§7.1).

Validated invocation (reference):

    python -m mlx_lm lora --model models/EXAONE-3.5-2.4B-Instruct-bf16 \
        --train --data data/mlx -c configs/sft.yaml

Phase 2 adds: assistant-turn masking (mask_prompt), capability-replay mixing
(§7.4b), and a JSONL TrainingCallback (§17.2 train metrics: loss/lr/grad_norm/
tok_per_s, checkpoint val_loss/J/acc_style/sim/ppl).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def mlx_lora_argv(config_path: str = "configs/sft.yaml") -> list[str]:
    """Build the mlx_lm.lora argv from the SFT config (pure, testable).

    The ``project:`` block in sft.yaml is project-level (read by the data build /
    callback), not an mlx key, so it is passed via ``-c`` but mlx ignores it. A
    clean temp-config split is added in Phase 2.
    """
    return [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train", "-c", str(config_path),
    ]


def run(config_path: str = "configs/sft.yaml") -> None:
    """Full SFT driver with replay-mix + JSONL callback. Phase 2."""
    raise NotImplementedError(
        "Phase 2 — wrap mlx_lm.lora.run/train_model with a §17 TrainingCallback "
        "and §7.4b capability-replay mixing"
    )


if __name__ == "__main__":  # pragma: no cover
    print("Reference command:\n  " + " ".join(mlx_lora_argv()))
