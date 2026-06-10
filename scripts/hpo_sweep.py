"""HPO OFAT sweep on 7.8B (same v7 corpus -> val-min comparable). Waits for v7, runs
configs sequentially, keeps only each config's val-min adapter (disk-safe)."""
import subprocess, re, time, os, glob, json, shutil
from pathlib import Path
v7pid=int(open("logs/train/sft_v7.pid").read())
print(f"waiting for v7 (pid {v7pid}) to free GPU...", flush=True)
while True:
    try: os.kill(v7pid,0); time.sleep(60)
    except OSError: break
print("v7 done -> starting HPO sweep", flush=True)
BASE=open("configs/sft.yaml").read(); ITERS=2000
CONFIGS=[("r8",8,"8.0","5.0e-5"),("r32",32,"32.0","5.0e-5"),("lr2e5",16,"16.0","2.0e-5"),("lr1e4",16,"16.0","1.0e-4")]
results=[]
for name,rank,scale,lr in CONFIGS:
    y=BASE.replace("  rank: 16",f"  rank: {rank}").replace("  scale: 16.0",f"  scale: {scale}") \
          .replace("learning_rate: 5.0e-5",f"learning_rate: {lr}") \
          .replace("  arguments: [5.0e-5, 2618, 1.0e-6]",f"  arguments: [{lr}, {ITERS}, 1.0e-6]") \
          .replace("iters: 2618",f"iters: {ITERS}").replace('adapter_path: "adapters/sft"',f'adapter_path: "adapters/hpo_{name}"')
    yp=f"/tmp/hpo_{name}.yaml"; open(yp,"w").write(y)
    ad=Path(f"adapters/hpo_{name}"); ad.mkdir(exist_ok=True)
    for f in glob.glob(f"{ad}/*"): os.remove(f)
    log=f"logs/train/hpo_{name}.log"
    print(f"\n=== TRAIN {name}: rank{rank} scale{scale} lr{lr} @ {ITERS}it  ({time.strftime('%H:%M')}) ===",flush=True)
    with open(log,"w") as lf:
        subprocess.run([".venv/bin/python","-u","-m","mlx_lm","lora","--model","models/EXAONE-3.5-7.8B-Instruct-bf16","--train","--data","data/mlx","-c",yp],stdout=lf,stderr=subprocess.STDOUT)
    vals=[(int(a),float(b)) for a,b in re.findall(r"Iter (\d+): Val loss ([0-9.]+)",open(log,errors="ignore").read())]
    if vals:
        bi,bv=min(vals,key=lambda x:x[1])
        ck=f"{ad}/{bi:07d}_adapters.safetensors"
        if os.path.exists(ck): shutil.copy(ck,f"{ad}/adapters.safetensors")
        for f in glob.glob(f"{ad}/0*_adapters.safetensors"): os.remove(f)
        results.append({"name":name,"rank":rank,"scale":float(scale),"lr":lr,"val_min":bv,"iter":bi})
        print(f"  -> val-min {bv:.3f}@it{bi} (kept, cleaned)",flush=True)
    json.dump(results,open("logs/eval/hpo_results.json","w"),ensure_ascii=False,indent=1)
print("\n=== HPO SWEEP DONE — ranked by val-min ===",flush=True)
for r in sorted(results,key=lambda x:x["val_min"]): print(f"  {r['name']:7} val {r['val_min']:.3f}  (rank{r['rank']} scale{r['scale']} lr{r['lr']})",flush=True)
