"""Record the Phase 0 MLX verification-gate (§7.1) results to JSONL (§17).

This both (a) demonstrates the logging skeleton works end-to-end and (b) leaves
a reproducible record of the gate outcome. Values are the measured results of
the convert + 1-step(×2) LoRA smoke test run during Phase 0 bootstrap.

Run:  .venv/bin/python scripts/log_phase0_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from logging_utils import JsonlLogger, new_run_id, write_manifest, iter_records  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "uncommitted"


def main() -> None:
    run_id = new_run_id("phase0-smoke")

    write_manifest(
        run_id,
        git_commit=_git_commit(),
        config_snapshot="configs/sft.yaml",
        seed=42,
        base_model="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        dataset_version="smoke-gold-8",
        dtype="bfloat16",
        hardware="Apple M4 Max 36GB",
        extra={
            "toolchain": "mlx-lm==0.29.1 + transformers==4.57.6 (pinned, §7.1 fallback)",
            "python": "3.12.12",
        },
    )

    log = JsonlLogger(phase="train", run_id=run_id, stage="sft")

    # §7.1 gate sub-results --------------------------------------------------- #
    log.event(
        "convert",
        metrics={"ok": True, "model_gb": 4.5},
        meta={"dtype": "bfloat16", "tool": "mlx_lm.convert"},
    )
    log.event(
        "smoke_load_generate",
        metrics={"ok": True, "tokens_per_sec": 73.3, "peak_mem_gb": 4.882},
        meta={"path": "models/EXAONE-3.5-2.4B-Instruct-bf16"},
    )
    log.event(
        "lora_attach",
        metrics={
            "trainable_pct": 0.061,
            "trainable_params_m": 1.466,
            "total_params_m": 2405.327,
        },
        meta={"fine_tune_type": "lora", "num_layers": 4},
    )
    # the two smoke training steps
    log.step(1, train_loss=7.173, val_loss=7.026, lr=1.0e-5, tok_per_s=41.321, peak_mem_gb=4.945)
    log.step(2, train_loss=6.703, val_loss=6.160, lr=1.0e-5, tok_per_s=182.251, peak_mem_gb=5.092)

    log.event(
        "gate_result",
        metrics={"passed": True, "loss_decreasing": True, "adapter_saved": True},
        meta={
            "section": "§7.1",
            "note": "EXAONE-3.5 trains under mlx-lm; pinned transformers<5 for "
            "EXAONE remote-config compatibility.",
        },
    )

    print(f"[ok] wrote {log.path}")
    n = sum(1 for _ in iter_records(log.path))
    print(f"[ok] {n} event records; manifest at runs/manifest.jsonl")


if __name__ == "__main__":
    main()
