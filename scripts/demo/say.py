import json, sys, time
from pathlib import Path
CHAT = Path(__file__).resolve().parents[2]/"logs/chat"
INBOX, OUTBOX = CHAT/"inbox.jsonl", CHAT/"outbox.jsonl"
text = sys.argv[1]
ids = [json.loads(l)["id"] for l in INBOX.read_text(encoding="utf-8").splitlines() if l.strip()]
nid = (max(ids)+1) if ids else 0
with INBOX.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"id":nid,"text":text}, ensure_ascii=False)+"\n")
for _ in range(700):
    for l in OUTBOX.read_text(encoding="utf-8").splitlines():
        if l.strip() and json.loads(l)["id"] == nid:
            print(json.loads(l)["response"]); sys.exit(0)
    time.sleep(0.5)
print("[timeout]"); sys.exit(1)
