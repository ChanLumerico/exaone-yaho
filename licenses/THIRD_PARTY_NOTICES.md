# Third-Party Notices

This project (non-commercial, research/portfolio) builds on the following. Each is
governed by its own license; users must comply with all of them.

## Base model
- **EXAONE-3.5-2.4B-Instruct** — LG AI Research.
  License: **EXAONE AI Model License Agreement 1.1 - NC** (research/non-commercial).
  Full text bundled at [`EXAONE_AI_Model_License.txt`](EXAONE_AI_Model_License.txt).
  Obligations honored: derivative name starts with "EXAONE"
  (`EXAONE-3.5-2.4B-Instruct-Yaho`); license copy shipped in repo + HF; outputs are
  non-commercial; base + license stated in the model card.

## Datasets (seed corpora)
- **SmileStyle** (`smilegate-ai/korean_smile_style_dataset`) — Smilegate AI.
  License: **CC-BY-NC 4.0** (attribution, non-commercial). Primary multi-turn seed.
- **AI Hub** 일상/감성 대화 (optional) — NIA; account + approval required, per-dataset terms.
- **모두의 말뭉치** (optional) — National Institute of Korean Language; application required.
- **NSMC** (Naver sentiment movie corpus) — CC0 / public; minor single-turn seasoning +
  negativity-scorer training.
- **KLUE-YNAT** — CC-BY-SA 4.0; ~10% formal single-turn seasoning.

## Auxiliary models
- **Stylizer (bulk):** Qwen2.5-/Qwen3-32B-class — **Apache-2.0** (clean output license).
- **Sentiment / NLI scorers (offline teachers, §5.3/§5.4):**
  - `Copycats/koelectra-base-v3-generalized-sentiment-analysis`
  - `rkdaldus/ko-sent5-classification`
  - a Korean zero-shot NLI model (departure scorer)
  See each model's HF card for its license; all used non-commercially.

## This project's code
- Licensed **Apache-2.0** (see top-level `LICENSE`). Note: the *code* license is
  permissive, but the *model weights and their outputs* are bound by the EXAONE
  License 1.1-NC above — i.e., the overall artifact is **non-commercial**.

## Persona note (§14)
Inspired by gyaru culture and the RESCENE/Minami meme lineage. This is a gyaru *style*
model, **not** an impersonation of, or endorsed by, any real individual or group. No real
person's content is reproduced; training data is synthesized.
