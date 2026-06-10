"""Interactive-ish driver: load v12.2 once, fill in assistant responses for any user turn
that lacks one across multiple conversations. Lets me append my next (reactive) turns to the
JSON and re-run. Fixed lx.SYSTEM_PROMPT anchor / temp 0.8. Usage: python chat_drive.py <adapter> <conv.json>"""
import sys, json
sys.path.insert(0, "src")
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import lexicon as lx

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "adapters/sft_v122_orpo"
CONV = sys.argv[2] if len(sys.argv) > 2 else "data/_chat.json"
print(f"loading {ADAPTER} ...", flush=True)
model, tok = load("models/EXAONE-3.5-7.8B-Instruct-bf16", adapter_path=ADAPTER)
sampler = make_sampler(temp=0.8, top_p=0.9)
SYS = {"role": "system", "content": lx.SYSTEM_PROMPT}

convos = json.load(open(CONV))  # list of conversations; each = list of {"role","content"}
seed = 7
for ci, conv in enumerate(convos):
    msgs = [SYS]
    print(f"\n{'='*60}\n[대화 {ci+1}]")
    i = 0
    while i < len(conv):
        m = conv[i]
        if m["role"] == "user":
            msgs.append(m)
            print(f"🧑 {m['content']}")
            # generate response if next isn't already an assistant turn
            if i + 1 >= len(conv) or conv[i + 1]["role"] != "assistant":
                mx.random.seed(seed); seed += 1
                r = generate(model, tok, prompt=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                             max_tokens=260, sampler=sampler, verbose=False).strip()
                conv.insert(i + 1, {"role": "assistant", "content": r})
            msgs.append(conv[i + 1])
            print(f"🎀 {conv[i+1]['content']}")
            i += 2
        else:
            msgs.append(m); print(f"🎀 {m['content']}"); i += 1

json.dump(convos, open(CONV, "w"), ensure_ascii=False, indent=1)
print("\n[saved]", CONV, flush=True)
