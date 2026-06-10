"""Custom ORPO (Hong et al. 2024) in MLX — corrective preference tune of v8-r16.
L = L_SFT(chosen) + λ·(-log σ(log_odds(chosen) - log_odds(rejected))). Starts from the
v8-r16 LoRA adapter (mlx_lm has no ORPO; we replicate its LoRA setup + custom loss/loop).
  pairs: data/orpo_pairs.jsonl   out: adapters/sft_v8_orpo
  smoke: python scripts/orpo_train.py --smoke   (10 iters, validate mechanics)
"""
import sys, json, os, time, random, shutil
sys.path.insert(0,"src")
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.utils import load
from mlx_lm.tuner.utils import linear_to_lora_layers
import lexicon as lx

SMOKE = "--smoke" in sys.argv
def _arg(f,d):
    return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
BASE="models/EXAONE-3.5-7.8B-Instruct-bf16"; START=_arg("--start","adapters/sft_v8_r16/adapters.safetensors")
OUT=_arg("--out","adapters/sft_v8_orpo"); LAM=0.3; LR=1.0e-5; ITERS=10 if SMOKE else int(_arg("--iters","363"))  # ~3 epochs over 121 (override for larger pair sets)

print("loading base + v8-r16 LoRA...", flush=True)
model, tok = load(BASE)
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank":16,"scale":16.0,"dropout":0.0})
model.load_weights(START, strict=False)
mx.eval(model.parameters())
ntrain=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
print(f"trainable LoRA params: {ntrain/1e6:.2f}M", flush=True)

pairs=[json.loads(l) for l in open(_arg("--pairs","data/orpo_pairs.jsonl"),encoding="utf-8") if l.strip()]
def encode(user,resp):
    pm=[{"role":"system","content":lx.SYSTEM_PROMPT},{"role":"user","content":user}]
    pids=tok.apply_chat_template(pm, add_generation_prompt=True)
    fids=tok.apply_chat_template(pm+[{"role":"assistant","content":resp}], add_generation_prompt=False)
    return fids, len(pids)
enc=[(*encode(p["prompt"],p["chosen"]), *encode(p["prompt"],p["rejected"])) for p in pairs]
enc=[e for e in enc if (len(e[0])-e[1])>=3 and (len(e[2])-e[3])>=3 and len(e[0])<=1100 and len(e[2])<=1100]
print(f"encoded {len(enc)} pairs (len-capped 1100)", flush=True)

def resp_logp(model, ids, start):
    x=mx.array(ids)[None]
    logp=nn.log_softmax(model(x[:,:-1]).astype(mx.float32), axis=-1)[0]
    tl=logp[mx.arange(len(ids)-1), mx.array(ids[1:])]
    r=tl[start-1:]
    return r.sum(), r.shape[0]
def loss_fn(model, idxs):
    tot=mx.array(0.0)
    for i in idxs:
        cf,cs,rf,rs=enc[i]
        sw,nw=resp_logp(model,cf,cs); sl,nl=resp_logp(model,rf,rs)
        lw_r=sw/nw; ll_r=sl/nl
        lw=mx.minimum(lw_r,-0.05); ll=mx.minimum(ll_r,-0.05)   # odds 항 그래디언트 폭발 방지(lw->0 특이점)
        L_or=-nn.log_sigmoid((lw-mx.log(-mx.expm1(lw)))-(ll-mx.log(-mx.expm1(ll))))  # log(1-exp) 안정형
        tot=tot + (-lw_r) + LAM*L_or
    return tot/len(idxs)

lg=nn.value_and_grad(model, loss_fn)
opt=optim.AdamW(learning_rate=LR)
rng=random.Random(42); order=list(range(len(enc))); rng.shuffle(order)
t0=time.time()
for it in range(1,ITERS+1):
    idxs=[order[(it-1)%len(order)]]
    loss,grads=lg(model,idxs)
    opt.update(model,grads); mx.eval(model.parameters(),opt.state)
    if it%(2 if SMOKE else 30)==0 or it==1:
        print(f"iter {it}/{ITERS}  loss {float(loss):.4f}  ({(time.time()-t0)/it:.1f}s/it)", flush=True)
if not SMOKE:
    os.makedirs(OUT,exist_ok=True)
    mx.save_safetensors(os.path.join(OUT,"adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    shutil.copy(os.path.join(os.path.dirname(START),"adapter_config.json"), os.path.join(OUT,"adapter_config.json"))
    open(os.path.join(OUT,"SELECTED.txt"),"w").write(f"ORPO of v8-r16: λ={LAM} lr={LR} iters={ITERS} (crisis+yaho+para pairs)")
    print(f"[ok] saved {OUT}", flush=True)
else:
    print("[smoke ok] mechanics validated", flush=True)
