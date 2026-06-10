import sys, json, math, os
sys.path.insert(0,"src")
from mlx_lm import load
import mlx.core as mx, mlx.nn as nn
from datasets import load_dataset
TAG, ADAPTER = sys.argv[1], sys.argv[2]
N=int(sys.argv[3]) if len(sys.argv)>3 else 200
items=list(load_dataset("skt/kobest_v1","copa",split="test"))[:N]
CONN={"결과":"그래서","원인":"왜냐하면"}
def slp(model,tok,p,c):
    pids=tok.encode(p); fids=tok.encode(p+c); cont=fids[len(pids):]
    if not cont: return -1e9
    lg=model(mx.array(fids)[None][:,:-1])[0].astype(mx.float32); lp=nn.log_softmax(lg,axis=-1); st=len(pids)-1
    return float(lp[st:st+len(cont)][mx.arange(len(cont)), mx.array(cont)].sum())
model,tok=load("models/EXAONE-3.5-7.8B-Instruct-bf16", adapter_path=ADAPTER)
c=0
for it in items:
    q=CONN.get(it["question"],"그래서"); pr=it["premise"]+" "+q+" "
    if (0 if slp(model,tok,pr,it["alternative_1"])>slp(model,tok,pr,it["alternative_2"]) else 1)==it["label"]: c+=1
acc=c/len(items)
tl=tn=0.0
for it in items[:120]:
    ids=tok.encode(it["premise"]+" "+it["alternative_1"])
    if len(ids)<2: continue
    lg=model(mx.array(ids)[None][:,:-1])[0].astype(mx.float32); lp=nn.log_softmax(lg,axis=-1)
    tl+=float(lp[mx.arange(len(ids)-1), mx.array(ids[1:])].sum()); tn+=len(ids)-1
ppl=math.exp(-tl/max(tn,1))
cm=json.load(open("logs/eval/cap_metrics.json")) if os.path.exists("logs/eval/cap_metrics.json") else {}
cm[TAG]={"copa_acc":acc,"neutral_ppl":ppl}
json.dump(cm, open("logs/eval/cap_metrics.json","w"), ensure_ascii=False, indent=1)
print(f"[{TAG}] COPA acc={acc:.3f} | ppl={ppl:.2f}")
