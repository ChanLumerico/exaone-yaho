"""Inference — STANDARD generate() ONLY (§5.6, LOCKED).

The deployed artifact is the fine-tuned weights alone. The model has internalized
the yaho/parapara policy, noun extraction, and arbitration (§5.5). There is NO
external scorer, morphological analyzer, or gating logic here — wiring any in
would prevent §8/§16.3 from measuring the *model's learned* behavior.

This is a thin multi-turn chat wrapper over mlx_lm. The same fixed system prompt
used in training is used here (§7.4a — train==infer anchor). Generation
hyperparameters default to the serving block of configs/deploy.yaml (§11 — no
hardcoded hyperparameters); explicit args override.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT = "너는 갸루 말투로 대답하는 AI야."


def _serving_config(config_path: str = "configs/deploy.yaml") -> dict:
    import yaml

    p = Path(config_path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open(encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("serving", {})


def chat(
    messages: list[dict[str, str]],
    *,
    model_path: str = "models/EXAONE-3.5-7.8B-Instruct-bf16",
    adapter_path: str | None = "adapters/sft",
    config_path: str = "configs/deploy.yaml",
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate the next gyaru assistant turn for a multi-turn ``messages`` list.

    Prepends the fixed system prompt if absent. Standard generate only (§5.6).
    ``max_tokens`` / ``temperature`` / ``top_p`` fall back to deploy.yaml when None.
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    cfg = _serving_config(config_path)
    max_tokens = cfg.get("max_tokens", 512) if max_tokens is None else max_tokens
    temperature = cfg.get("temperature", 0.8) if temperature is None else temperature
    top_p = cfg.get("top_p", 0.9) if top_p is None else top_p

    if not messages or messages[0].get("role") != "system":
        system = cfg.get("system_prompt", SYSTEM_PROMPT)
        messages = [{"role": "system", "content": system}, *messages]

    model, tokenizer = load(
        model_path,
        adapter_path=adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    sampler = make_sampler(temp=temperature, top_p=top_p)
    return generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
    )
