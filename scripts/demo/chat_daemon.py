"""Resident v6 chat daemon — model loaded ONCE in the MAIN thread so the GPU
actually engages (mlx_lm.server runs generation in a worker thread and dies with
'no Stream(gpu,0) in current thread'). File-queue protocol, persists transcript.
  inbox.jsonl : {"id":N,"text":"<user>"}    (driver appends one per turn)
  outbox.jsonl: {"id":N,"response":"<v6>"}   (daemon appends)
  {"text":"__RESET__"} clears the conversation.  daemon.ready marks model loaded.
"""
import json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT/"logs/chat"; CHAT.mkdir(parents=True, exist_ok=True)
INBOX, OUTBOX, STATE, READY = CHAT/"inbox.jsonl", CHAT/"outbox.jsonl", CHAT/"v6_state.json", CHAT/"daemon.ready"
cfg = yaml.safe_load((ROOT/"configs/deploy.yaml").read_text(encoding="utf-8"))["serving"]
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os
_adapter=os.environ.get("YAHO_ADAPTER", cfg["adapter_path"])
print("[adapter]", _adapter, flush=True)
model, tok = load(cfg["model_path"], adapter_path=_adapter, tokenizer_config={"trust_remote_code": True})
sampler = make_sampler(temp=cfg["temperature"], top_p=cfg["top_p"])
SYS = {"role":"system","content":cfg["system_prompt"]}
msgs = [SYS]
for p in (INBOX, OUTBOX): p.write_text("", encoding="utf-8")
STATE.write_text(json.dumps(msgs, ensure_ascii=False, indent=1), encoding="utf-8")
READY.write_text("ready"); seen = 0
while True:
    lines = [l for l in INBOX.read_text(encoding="utf-8").splitlines() if l.strip()]
    for line in lines[seen:]:
        req = json.loads(line)
        if req.get("text") == "__RESET__":
            msgs = [SYS]
        else:
            msgs.append({"role":"user","content":req["text"]})
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            resp = generate(model, tok, prompt=prompt, max_tokens=512, sampler=sampler, verbose=False).strip()
            msgs.append({"role":"assistant","content":resp})
            with OUTBOX.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id":req["id"],"response":resp}, ensure_ascii=False)+"\n")
            STATE.write_text(json.dumps(msgs, ensure_ascii=False, indent=1), encoding="utf-8")
    seen = len(lines)
    time.sleep(0.4)
