---
base_model: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
language: [ko]
license: other
license_name: exaone-ai-model-license-1.1-nc
license_link: licenses/EXAONE_AI_Model_License.txt
tags: [korean, gyaru, style-transfer, llm-alignment, mlx, exaone, non-commercial]
pipeline_tag: text-generation
---

# EXAONE-3.5-2.4B-Instruct-Yaho (model card)

> ⚠️ Status: **in development (Phase 0 complete).** No fine-tuned weights are published
> yet. This card describes the intended artifact.

## Model summary
A LoRA fine-tune of `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` that responds in a Korean
**gyaru (갸루)** persona and exhibits two learned behavioral policies — **yaho**
(deflecting negativity with `{noun} 야호~`) and **parapara** (filling departure/idle
turns). It is a **style + behavioral-policy alignment** study, not a knowledge model.

- **Base:** EXAONE-3.5-2.4B-Instruct (Korean/English, 2.4B).
- **Method:** bf16 plain LoRA SFT via mlx-lm on Apple Silicon; optional ORPO/DPO later.
- **Inference:** standard `generate()` only — the model internalizes the policy; there
  is no external scorer, gate, or morphological analyzer at serving time.

## Intended use
Research, education, and portfolio demonstration of controllable style alignment.
Free, non-commercial multi-turn chat demo only.

## Out of scope / limitations
- **Non-commercial only** (EXAONE License 1.1-NC). No paid endpoints/APIs.
- A small 2.4B stylized model: expect reduced factuality and occasional over/under-
  styling. Capability-retention vs. the base is measured and reported (§16.1).
- Korean-centric; gyaru slang is niche and may be unfamiliar to general users.
- May deflect (yaho) where a literal helpful answer was expected — by design; tuned to
  also actually address user intent (§15 checklist).

## Persona & ethics
The persona is **inspired by** gyaru culture (and the RESCENE/Minami meme lineage) but
is a **gyaru *style* model, not an impersonation** of any real individual (§14). No real
person's utterances/content are scraped; training data is **synthesized** (§6).

## Training data
- **Seed (structure/content only):** SmileStyle (`smilegate-ai/korean_smile_style_dataset`,
  **CC-BY-NC 4.0**), optionally AI Hub / 모두의 말뭉치 (multi-turn); NSMC/KLUE-YNAT as
  minor single-turn seasoning + scorer training. *Gyaru-ness is added by our pipeline,
  not taken from the seeds.*
- **Stylizer:** frontier model for ~30–50 gold anchors; local Qwen-32B-class (Apache-2.0)
  for bulk — clean output license (§16.4).

## Evaluation (planned, §8/§16)
Style accuracy, meaning preservation, multi-turn consistency, yaho/parapara firing-curve
fidelity, SFT-vs-preference win-rate, perplexity, J = Acc_style·Sim, and capability
retention — all on a fixed seed/eval-set with reproducible JSONL-backed reports.

## License
Weights/outputs: **EXAONE AI Model License 1.1-NC** (`licenses/EXAONE_AI_Model_License.txt`).
Code: Apache-2.0. Third-party data/models: `licenses/THIRD_PARTY_NOTICES.md`.
