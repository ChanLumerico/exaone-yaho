# exaone-yaho 🌟

> Aligning **EXAONE-3.5-2.4B-Instruct** to a Korean **gyaru (갸루)** persona.
> A meme on the surface; a real LLM-alignment case study underneath:
> **data synthesis → SFT → (optional) preference optimization → evaluation → deployment.**
> **Non-commercial, research/portfolio only.**

The model speaks in high-tension gyaru style (Japanese-transliterated interjections +
emoji + kaomoji) and learns two behavioral policies:
- **야호 (yaho)** — deflects negativity/pressure with `{핵심명사} 야호~`
  (e.g. *"...거제 시민들한테 혼나"* → *"거제 야호~✨"*).
- **파라파라 (parapara)** — fills departure/idle/farewell turns with
  `네가 {활동}하는 동안, 그럼 난 파라파라나 추고있어야겠다~💖`.

These are taught **at data-synthesis time** by dense, convex-gated offline "teacher"
pipelines. At inference there is **only the fine-tuned model** — no scorers, no gating,
no morphological analysis (it internalizes the policy). See the design brief for full intent.

## Status
**Phase 0 (Foundation) complete** — the EXAONE↔mlx-lm training path is verified on
Apple Silicon (convert → LoRA train → adapter inference), logging + tests in place.
**Next: Phase 1 (gyaru corpus synthesis).** See [CHANGELOG.md](CHANGELOG.md) and
[CLAUDE.md](CLAUDE.md).

## Setup (Apple Silicon, MLX-first)
```bash
uv venv --python 3.12 .venv
uv pip install -e ".[data,analysis,dev]"   # core toolchain is pinned (see below)

# convert the base model to bf16 MLX (one-time)
python -m mlx_lm convert --hf-path LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
  --mlx-path models/EXAONE-3.5-2.4B-Instruct-bf16 --dtype bfloat16 --trust-remote-code

python -m pytest -q
```

> **Pinned toolchain:** `mlx-lm==0.29.1` + `transformers==4.57.6`. EXAONE-3.5's remote
> config is transformers-4.x-era and breaks under transformers ≥5.0 (required by
> mlx-lm ≥0.31). Details in [CLAUDE.md](CLAUDE.md).

## Layout
```
configs/   style/gate/sft/dpo/deploy YAML — all hyperparameters (§11)
src/       lexicon, gating, pipelines (teachers), stylizer, build, train, eval, infer, logging
data/      seed/ gold/ + synthesized corpora        logs/  append-only JSONL (SSOT, §17)
runs/      manifest.jsonl (reproducibility)         analysis/ JSONL→figures
reports/   eval reports        space/ Gradio demo   licenses/ EXAONE + third-party
```

## Hardware
Developed on Apple **M4 Max, 36GB**, macOS. CUDA-only tooling (Unsloth, bitsandbytes
4bit, flash-attention) is **not** used.

## License & attribution
- **Code:** Apache-2.0 (`LICENSE`).
- **Base model & fine-tuned weights:** **EXAONE AI Model License 1.1-NC** —
  research/**non-commercial** only; derivative name starts with "EXAONE"; license copy
  bundled (`licenses/EXAONE_AI_Model_License.txt`). See [MODEL_CARD.md](MODEL_CARD.md).
- **Data:** SmileStyle (CC-BY-NC 4.0) and others — see `licenses/THIRD_PARTY_NOTICES.md`.
- Persona is *inspired by* gyaru culture; **not** an impersonation of any real person (§14).
