"""Local MLX gyaru chat demo (§5.6 — standard generate ONLY; no scorer/gating).

Loads the base 7.8B MLX model + the production LoRA adapter (deploy.yaml) once and
serves a multi-turn Gradio chat. The system prompt is the EXACT training anchor
(deploy.yaml:serving.system_prompt = lx.SYSTEM_PROMPT) so infer==train (§7.4a).

Install:  uv pip install gradio
Run:      .venv/bin/python scripts/demo/app.py            # http://127.0.0.1:7860
          .venv/bin/python scripts/demo/app.py --share    # public gradio link
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
NON_COMMERCIAL = ("⚠️ 비상업·연구/포트폴리오 전용 (EXAONE 라이선스 §14). 밈 페르소나 데모이며 "
                  "실제 상담/의료/법률 조언이 아닙니다. 위기 시 자살예방 상담 ☎️109.")


def _cfg() -> dict:
    return (yaml.safe_load((ROOT / "configs/deploy.yaml").read_text(encoding="utf-8")) or {})["serving"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--adapter", default=None, help="override serving.adapter_path")
    a = ap.parse_args()

    import gradio as gr
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    cfg = _cfg()
    system = cfg["system_prompt"]
    sampler = make_sampler(temp=cfg.get("temperature", 0.8), top_p=cfg.get("top_p", 0.9))
    max_tokens = cfg.get("max_tokens", 768)
    print(f"[load] {cfg['model_path']} + adapter {a.adapter or cfg['adapter_path']} …")
    model, tok = load(cfg["model_path"], adapter_path=a.adapter or cfg["adapter_path"],
                      tokenizer_config={"trust_remote_code": True})
    print("[ready] http://127.0.0.1:%d" % a.port)

    def respond(message, history):
        msgs = [{"role": "system", "content": system}]
        for h in history:                                  # gradio messages format
            if isinstance(h, dict):
                msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": message})
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = ""
        for r in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
            out += r.text
            yield out

    gr.ChatInterface(
        respond, type="messages",
        title="EXAONE-3.5-7.8B-Instruct-Yaho 🎀  (갸루 페르소나 데모)",
        description=NON_COMMERCIAL,
        examples=["오늘 시험 망쳤어…", "주말에 뭐하지?", "퇴근하고 집 갈게", "트랜스포머 어텐션 원리 설명해줘"],
    ).launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
