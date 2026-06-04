# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org).

## [Unreleased] — Phase 1 (gyaru corpus)
### Added
- **`PERSONA.md`** — persona SSOT: refined **languid '갸루귀신'** (낮은 텐션·나른함, 손해 logic,
  퀸/마이웨이 mindset, 타격감 제로, MZ slang, 설렁탕 reversal) with diverse examples + prompt anchor.
- **Phase 1d synthesis** — Qwen3-30B-A3B-Instruct MLX teacher (`src/stylizer.py`), PersonaChat-Korean
  (MIT) multi-turn seed, `src/data_build.py` (stylize→arbiter→decorate→filter), §5.2 katakana light pass.
### Changed
- **Persona refined to languid '갸루귀신'** (corrects original "하이텐션"). Gold **regenerated** as 50
  fresh shots in the new tone (5-agent workflow, balanced yaho 44%/para 34%/neutral 30%, 35% multiturn,
  0 §5.5 violations); stylizer prompts + `refined_fewshot.jsonl` + lexicon (🙄✌️ + 손해/퀸/MZ vocab) updated.
### Added (earlier this phase)
- **Gold set** (`data/gold/gold.jsonl`, 50 dialogues): user-handcrafted, then
  frontier-refined (`scripts/enhance_gold.py`) — §5.5 arbitration fixes, 34%
  multiturn, marker diversity 32/39, §5.2 katakana (p_kata=0.12, seeded). Adversarial
  3-reviewer pass. Tooling: `validate_gold.py`, `gold_split.py` (§16.4 ①=16 / meme=34).
- **Phase 1a — §5.2 decorator** (`src/decorator.py`): `decorate()` (rules 1–5:
  interjection inject, katakana, boundary emoji/kaomoji, hi-tension endings, tails)
  + `postprocess()` (rule 6: collapse tail/emoji runs, tidy) + `DecoratorParams`
  (loaded from `configs/style.yaml`, §11) + `Decorator` (λ sampling). Mechanical
  categories only; context-only categories left to the stylizer (§5.2 note). 11 tests.

## [0.1.0] - 2026-06-04
Phase 0 — Foundation. The repo boots and the EXAONE↔mlx-lm training path is proven.

### Added
- §9 directory skeleton, `git` repo, packaging (`pyproject.toml`, `requirements.txt`,
  `.gitignore`, `.env.example`).
- `configs/{style,gate,sft,dpo,deploy}.yaml` — all probabilities/hyperparameters
  externalized (§11), seeded with spec defaults.
- `src/lexicon.py` (§5.1 data, SSOT), `src/gating.py` (§5.3/5.4/5.5 convex gate +
  arbitration — realized & unit-tested), and interface ABCs + Phase-1 stubs for the
  stylizer, yaho/parapara pipelines, decorator, data/prefs build, train/eval/infer/serve.
- `src/logging_utils.py` — §17 append-only JSONL logging skeleton (`JsonlLogger`,
  run manifest, embedding index, analysis readers). Working.
- Tests: gating math, logging envelope, lexicon sanity — **18 passing**.
- Docs: `CLAUDE.md` (living memory), `README.md`, `MODEL_CARD.md`.
- Licenses: bundled **EXAONE AI Model License 1.1-NC** copy + third-party notices
  (SmileStyle CC-BY-NC, Qwen Apache-2.0, KoELECTRA), code `LICENSE` (Apache-2.0). (§14)
- **§7.1 MLX verification gate PASSED** and logged
  (`logs/train/phase0-smoke-*.jsonl`, `runs/manifest.jsonl`): convert → generate →
  2-step LoRA train (loss ↓ 7.17→6.70) → adapter inference.

### Changed
- **Toolchain pinned** to `mlx-lm==0.29.1` + `transformers==4.57.6` (Python 3.12):
  EXAONE-3.5's transformers-4.x-era remote config breaks under transformers ≥5.0
  (which mlx-lm ≥0.31 requires). This is the §7.1 fallback path, made canonical.
  *Why:* AutoTokenizer→AutoConfig fails on `rope_scaling`/`max_position_embeddings`
  under the new rope standardization; pinning is cleaner than patching vendored code.

### Fixed
- Post-review (adversarial 6-dimension Phase-0 conformance pass): logger now
  enforces numeric/bool metric values (§17.1); `infer.py` reads generation
  hyperparameters from `deploy.yaml` instead of hardcoding (§11); CHANGELOG smoke
  wording clarified. Review found no blockers/majors.

### Notes
- Preference optimization (DPO/ORPO) and `prefs.jsonl` remain **post-MVP** (§12 P2.5).
