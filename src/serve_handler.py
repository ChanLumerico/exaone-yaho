"""HF Inference Endpoint handler — STANDARD generate() ONLY (§5.6 / §10, LOCKED).

⚠️ Do NOT add a scorer, morphological analyzer (kiwipiepy), or yaho/parapara
gating logic here. Pre/post-processing is limited to chat-prompt formatting. The
model performs the policy itself; a wrapper would invalidate §8 / §16.3 ablations.

For local demo/eval prefer `mlx_lm.server` (OpenAI-compatible). This handler is
the transformers/TGI/vLLM path (Linux; needs HF-format weights, §7.5). Phase 4.
"""

from __future__ import annotations

SYSTEM_PROMPT = "너는 갸루 말투로 대답하는 AI야."


class EndpointHandler:  # HF Inference Endpoints contract
    def __init__(self, path: str = ""):
        # Phase 4: load HF-format (de-quantized) weights + tokenizer here.
        raise NotImplementedError("Phase 4 — load weights; standard generate only (§5.6)")

    def __call__(self, data: dict) -> list[dict]:
        # data["inputs"] = chat messages -> apply chat template -> generate -> text.
        raise NotImplementedError("Phase 4 — standard generate only, no gating (§5.6)")
