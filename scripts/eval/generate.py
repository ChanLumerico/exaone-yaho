"""Eval gen: each model config generates on the 50-probe set with the SHARED gyaru
system anchor (fair base-vs-SFT delta). Writes data/eval/gen_<tag>.jsonl."""
import sys, json, time
sys.path.insert(0, "src")
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import lexicon as lx

B24="models/EXAONE-3.5-2.4B-Instruct-bf16"; B78="models/EXAONE-3.5-7.8B-Instruct-bf16"
CONFIGS=[
 ("base24", B24, None),
 ("base78", B78, None),
 ("v2",     B24, "adapters/_v2_it90_backup"),
 ("v3",     B24, "adapters/sft_v3"),
 ("v4",     B78, "adapters/sft_v4"),
 ("v6",     B78, "adapters/sft"),
 ("v8",     B78, "adapters/sft_v8"),
 ("v8b",    B78, "adapters/sft_v8_r16"),
 ("orpo",   B78, "adapters/sft_v8_orpo"),
 ("orpo2",  B78, "adapters/sft_v8_orpo2"),
 ("v9",     B78, "adapters/sft_v9"),
 ("v9orpo", B78, "adapters/sft_v9_orpo"),
 ("v10",     B78, "adapters/sft_v10"),
 ("v10orpo", B78, "adapters/sft_v10_orpo"),
 ("v11",     B78, "adapters/sft_v11"),
 ("v11orpo", B78, "adapters/sft_v11_orpo"),
 ("v12",     B78, "adapters/sft_v12"),
 ("v12orpo", B78, "adapters/sft_v12_orpo"),
 ("v121orpo", B78, "adapters/sft_v121_orpo"),
 ("v122orpo", B78, "adapters/sft_v122_orpo"),
]
probes=[json.loads(l) for l in open("data/eval/probe.jsonl") if l.strip()]
only=set(sys.argv[1:])  # optional: subset of tags
for tag, base, adapter in CONFIGS:
    if only and tag not in only: continue
    t0=time.time(); print(f"[{tag}] loading {base} adapter={adapter}", flush=True)
    try: mx.set_cache_limit(3*1024**3)
    except Exception: pass
    model, tok = load(base, adapter_path=adapter)
    sampler=make_sampler(temp=0.7, top_p=0.9)
    out=Path(f"data/eval/gen_{tag}.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for i,p in enumerate(probes):
            msgs=[{"role":"system","content":lx.SYSTEM_PROMPT},{"role":"user","content":p["prompt"]}]
            prompt=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            mx.random.seed(42+i)  # reproducible per-probe
            r=generate(model, tok, prompt=prompt, max_tokens=400, sampler=sampler, verbose=False).strip()
            fh.write(json.dumps({**p,"response":r}, ensure_ascii=False)+"\n"); fh.flush()
            if (i+1)%10==0: print(f"  [{tag}] {i+1}/{len(probes)}", flush=True)
    del model, tok
    try: mx.clear_cache()
    except Exception: pass
    print(f"[{tag}] done {len(probes)} gens in {time.time()-t0:.0f}s -> {out}", flush=True)
print("[ALL DONE]", flush=True)
