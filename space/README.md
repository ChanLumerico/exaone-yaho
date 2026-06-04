# space/

Gradio multi-turn chat demo for Hugging Face **Spaces** (Phase 4, §10). Free tier,
non-commercial only (§14). Demonstrates the persona + both memes (yaho/parapara) in
multi-turn conversation.

⚠️ §5.6 (LOCKED): the demo calls standard `generate()` on the fine-tuned weights only —
**no** scorer, morphological analyzer, or gating logic. Spaces runs Linux (no MLX), so
serve from HF-format weights (see §7.5 export notes in `configs/deploy.yaml`).
