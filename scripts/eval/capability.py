"""Capability / catastrophic-forgetting eval: KoBEST-COPA accuracy (0-shot log-likelihood)
+ neutral-Korean perplexity. Per model. Writes logs/eval/cap_metrics.json."""
import sys, json, math
sys.path.insert(0,"src")
from mlx_lm import load
import mlx.core as mx, mlx.nn as nn
from datasets import load_dataset

B24="models/EXAONE-3.5-2.4B-Instruct-bf16"; B78="models/EXAONE-3.5-7.8B-Instruct-bf16"
CONFIGS=[("base24",B24,None),("base78",B78,None),("v2",B24,"adapters/_v2_it90_backup"),
         ("v3",B24,"adapters/sft_v3"),("v4",B78,"adapters/sft_v4"),("v5",B78,"adapters/sft")]
N_COPA=int(sys.argv[1]) if len(sys.argv)>1 else 250

copa=load_dataset("skt/kobest_v1","copa",split="test")
items=list(copa)[:N_COPA]
CONN={"결과":"그래서","원인":"왜냐하면"}

def seq_logprob(model, tok, prompt_text, cont_text):
    pids=tok.encode(prompt_text); fids=tok.encode(prompt_text+cont_text)
    cont=fids[len(pids):]
    if not cont: return -1e9, 0
    ids=mx.array(fids)[None]
    logits=model(ids[:, :-1])[0].astype(mx.float32)
    logp=nn.log_softmax(logits, axis=-1)
    start=len(pids)-1
    idx=mx.arange(len(cont))
    lp=logp[start:start+len(cont)][idx, mx.array(cont)]
    return float(lp.sum()), len(cont)

def eval_model(base, adapter):
    model, tok = load(base, adapter_path=adapter)
    # --- COPA ---
    correct=0
    for it in items:
        q=CONN.get(it["question"],"그래서")
        prompt=it["premise"]+" "+q+" "
        s1,_=seq_logprob(model,tok,prompt,it["alternative_1"])
        s2,_=seq_logprob(model,tok,prompt,it["alternative_2"])
        pred=0 if s1>s2 else 1
        if pred==it["label"]: correct+=1
    acc=correct/len(items)
    # --- neutral Korean perplexity (on COPA premises+alternatives = neutral text) ---
    tot_lp=tot_n=0.0
    for it in items[:120]:
        txt=it["premise"]+" "+it["alternative_1"]
        ids=tok.encode(txt)
        if len(ids)<2: continue
        arr=mx.array(ids)[None]
        logits=model(arr[:, :-1])[0].astype(mx.float32)
        logp=nn.log_softmax(logits,axis=-1)
        tgt=mx.array(ids[1:])
        lp=logp[mx.arange(len(ids)-1), tgt]
        tot_lp+=float(lp.sum()); tot_n+=len(ids)-1
    ppl=math.exp(-tot_lp/max(tot_n,1))
    del model, tok
    try: mx.clear_cache()
    except Exception: pass
    return acc, ppl

res={}
for tag,base,adapter in CONFIGS:
    print(f"[{tag}] COPA+ppl...", flush=True)
    acc,ppl=eval_model(base,adapter)
    res[tag]=dict(copa_acc=acc, neutral_ppl=ppl)
    print(f"  [{tag}] COPA acc={acc:.3f} | neutral ppl={ppl:.2f}", flush=True)
json.dump(res, open("logs/eval/cap_metrics.json","w"), ensure_ascii=False, indent=1)
print("-> logs/eval/cap_metrics.json")
