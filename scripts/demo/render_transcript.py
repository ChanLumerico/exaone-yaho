"""Render logs/chat/v6_state.json -> a human-readable .txt with 🧑 (me) / 🎀 (model)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
state = json.loads((ROOT/"logs/chat/v6_state.json").read_text(encoding="utf-8"))
out = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "logs/chat/v6_session.txt")
lines = ["EXAONE-3.5-7.8B-Instruct-Yaho (v6) — 라이브 1:1 대화 기록",
         "🧑 = 사람(평가자)   🎀 = 모델(v6)",
         "system prompt = train==infer 앵커 (lx.SYSTEM_PROMPT) / temp 0.8",
         "="*64, ""]
for m in state:
    if m["role"] == "system":
        continue
    tag = "🧑" if m["role"] == "user" else "🎀"
    lines.append(f"{tag}: {m['content']}")
    lines.append("")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"[ok] {out}  ({sum(1 for m in state if m['role']!='system')} turns)")
