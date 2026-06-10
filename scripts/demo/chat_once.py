"""One conversation turn against the resident v6 server. Persists multi-turn state.
Usage: python scripts/demo/chat_once.py "<user message>"   (reset: rm logs/chat/v6_state.json)"""
import sys, json, urllib.request
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "logs/chat/v6_state.json"
cfg = yaml.safe_load((ROOT/"configs/deploy.yaml").read_text(encoding="utf-8"))["serving"]
msgs = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else [{"role":"system","content":cfg["system_prompt"]}]
msgs.append({"role":"user","content":sys.argv[1]})
payload = {"messages":msgs,"temperature":cfg["temperature"],"top_p":cfg["top_p"],"max_tokens":cfg["max_tokens"],"stream":False}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
text = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
msgs.append({"role":"assistant","content":text})
STATE.write_text(json.dumps(msgs, ensure_ascii=False, indent=1), encoding="utf-8")
print(text)
