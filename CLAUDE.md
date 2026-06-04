# CLAUDE.md — exaone-yaho living memory

> Read this first every session. The handoff spec (the long brief) is the SSOT for
> *intent*; this file tracks *decisions, state, and gotchas*. Keep it current.

## What this is
Align **EXAONE-3.5-2.4B-Instruct** to a **gyaru persona** (Korean). Surface = meme;
substance = a real alignment pipeline: data synthesis → SFT → (optional) preference
opt → eval → deploy. **Non-commercial, portfolio/research only** (§14). GitHub
codename `exaone-yaho`; HF model name **`EXAONE-3.5-2.4B-Instruct-Yaho`** (must start
with "EXAONE", §14).

## Current status — Phase 0 COMPLETE ✅
The §7.1 MLX verification gate **passed** (see `logs/train/phase0-smoke-*.jsonl`):
convert → load+generate (73 tok/s) → LoRA attach (0.061% trainable) → 2-step train
(loss ↓ 7.17→6.70) → adapter applies in inference. Foundation skeleton + working
§17 logging + tests (18 passing) are in. **Next: Phase 1 (data corpus).**

## ⚠️ Key discovered decision — PINNED TOOLCHAIN (do not "upgrade")
EXAONE-3.5's remote config (`configuration_exaone.py`, `auto_map`, deprecated
`rope_scaling`) is **transformers-4.x-era** and **breaks under transformers ≥5.0**
(new rope standardization expects `rope_parameters`; AutoTokenizer→AutoConfig dies
with `'PreTrainedConfig' has no attribute 'max_position_embeddings'`). But
`mlx-lm ≥0.31` **hard-requires transformers ≥5.0**. Resolution: pin the newest
mlx-lm that still allows transformers<5.
- **VALIDATED: `mlx-lm==0.29.1` + `transformers==4.57.6`, Python 3.12.** Everything
  (convert, generate, lora, fuse, server) works; `mlx_lm.lora` passes
  `trust_remote_code=True` internally (lora.py:322), so no extra flag needed.
- transformers 5.x has native `exaone4`/`exaone_moe` but **not** plain `exaone` (3.5).
- If you must move to mlx-lm ≥0.31 later: you'd have to make the model self-contained
  (strip `auto_map`, migrate `rope_scaling`→`rope_parameters`, register `exaone` config),
  which the CLIs can't pick up cross-process. Not worth it; the pin is the clean path.

## Other decisions & rationale
- **Python 3.12** for `.venv` (uv), not the system's 3.14 — Phase-1 stack (kiwipiepy,
  sentence-transformers) lags on 3.14. Core MLX stack works on 3.12.
- **Base: EXAONE-3.5-2.4B, bf16 (full), plain LoRA** (not QLoRA) — 36GB has headroom,
  fidelity first (§7.1). Converted via `mlx_lm.convert --dtype bfloat16 --trust-remote-code`
  → `models/EXAONE-3.5-2.4B-Instruct-bf16` (4.5GB, gitignored).
- **§5.6 LOCKED — inference = fine-tuned weights ONLY.** Pipelines/scorers/kiwi are
  OFFLINE TEACHERS for Phase-1 data synthesis, discarded after training. NEVER put a
  scorer/morph-analyzer/gating into `infer.py`/`serve_handler.py`/Space — it would
  invalidate §8/§16.3 measurement of *learned* behavior.
- **`src/gating.py` is realized + tested** (pure §5.3/5.4/5.5 math). Scorers, extractors,
  decorator, pipelines, stylizer are **interface ABCs + Phase-1 stubs** (NotImplementedError).
- **Two-policy memes (never weaken, §5):** yaho = `{핵심명사} 야호~` deflects
  negativity/pressure; parapara = `네가 {활동}하는 동안 난 파라파라나 추고있어야겠다~💖`
  fills departure/idle. Dense convex gating (γ>1), not sparse lexicon branching.
  Arbitration §5.5: strong departure→parapara; never both in one response.

## MLX / Apple Silicon constraints (§7)
- M4 Max 36GB, MLX-first. **mlx-lm is the SFT engine.** PyTorch-MPS = fallback / future RL.
- ❌ FORBIDDEN: Unsloth, bitsandbytes 4bit QLoRA, flash-attention (all CUDA-only).
- Preference opt (DPO/ORPO) is **post-MVP** (§7.6, §12 P2.5); prefer ORPO (ref-free).

## Conventions
- All probabilities/hyperparams live in `configs/*.yaml` — **never hardcode** (§11).
- **Logging (§17):** every loggable signal → append-only JSONL via `src/logging_utils.py`
  (`JsonlLogger`), one file per run at `logs/<phase>/<run_id>.jsonl`; run row in
  `runs/manifest.jsonl`. Figures regenerate from JSONL (SSOT). **A step with no JSONL
  log is incomplete** — every Phase DoD includes its log.
- Fixed system prompt (train==infer anchor): `너는 갸루 말투로 대답하는 AI야.`
- Seeds fixed (42) for reproducibility. Tests accompany core logic.
- Added beyond §9 structure: `src/gating.py` (shared gate+arbitration), `src/logging_utils.py`,
  `scripts/smoke/` (smoke data+adapter), `licenses/`, `models/` (gitignored).

## Key hyperparameters (see configs/)
- LoRA: rank 16, scale(α) 32, dropout 0.05, lr 1e-4, seq 1024, all layers (`sft.yaml`).
- Gate yaho: p_base .05 / p_max .55 / γ 1.5 ; parapara: .04 / .60 / 1.4 (`gate.yaml`).
- Decorator: λ∈[0.2,0.6], P_kata .12, tail-after-marker .7 (`style.yaml`).
- Capability replay 10–20% (`sft.yaml project:`); early-stop on J=Acc_style·Sim (§7.4f).

## Validated commands
```bash
# activate
source .venv/bin/activate            # or prefix: .venv/bin/python ...

# convert base -> bf16 MLX (done; re-run only if needed)
python -m mlx_lm convert --hf-path LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
  --mlx-path models/EXAONE-3.5-2.4B-Instruct-bf16 --dtype bfloat16 --trust-remote-code

# generate (base or + adapter)
python -m mlx_lm generate --model models/EXAONE-3.5-2.4B-Instruct-bf16 \
  [--adapter-path adapters/sft] --prompt "..." --max-tokens 64 --temp 0.8

# SFT (Phase 2) — mlx_lm.lora handles trust_remote_code internally
python -m mlx_lm lora --model models/EXAONE-3.5-2.4B-Instruct-bf16 --train \
  --data data/mlx -c configs/sft.yaml

# tests
python -m pytest -q
```

## Known issues / TODO
- `mlx_lm.fuse` (Phase 4) output is MLX format; HF Inference Endpoint (TGI/vLLM) needs
  `--de-quantize` + MLX→HF conversion + **output-parity check** (§7.5). Untested yet.
- Phase 1 needs the data extras installed: `uv pip install -e ".[data,analysis]"`.
- Gold set: §A.4/§A.5 examples seeded into `scripts/smoke/data/` for the smoke test;
  Phase 1a must handcraft 30–50 gold dialogues into `data/gold/`.
- Harmless deprecation warning: `mx.metal.device_info` (mlx-lm 0.29.1 internal).
