"""Prefix-expand a MERGED gyaru corpus -> data/mlx (train/valid/test) for SFT.
Same seed-42 dialogue split + fixed lx.SYSTEM_PROMPT anchor as build_sft_dataset (v3+)."""
import json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/"src")); import lexicon as lx
src=ROOT/(sys.argv[1] if len(sys.argv)>1 else "data/gyaru_corpus_7.jsonl")
rng=random.Random(42)
rows=[json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
def prefix_expand(d):
    body=[m for m in d["messages"] if m["role"]!="system"]; sysm={"role":"system","content":lx.SYSTEM_PROMPT}; out=[]
    for i,m in enumerate(body):
        if m["role"]=="assistant" and m["content"].strip(): out.append({"messages":[sysm]+body[:i+1]})
    return out
rng.shuffle(rows)
n=len(rows); nval=max(8,n//20); ntest=max(8,n//20)
valid,test,train=rows[:nval],rows[nval:nval+ntest],rows[nval+ntest:]
def expand(sp):
    seen,res=set(),[]
    for d in sp:
        for ex in prefix_expand(d):
            k=json.dumps(ex,ensure_ascii=False)
            if k not in seen: seen.add(k); res.append(ex)
    return res
valid,test,train=expand(valid),expand(test),expand(train); rng.shuffle(train)
out=ROOT/"data/mlx"; out.mkdir(parents=True,exist_ok=True)
for name,sp in (("train",train),("valid",valid),("test",test)):
    (out/f"{name}.jsonl").write_text("\n".join(json.dumps(d,ensure_ascii=False) for d in sp)+"\n",encoding="utf-8")
lens=[sum(len(m["content"]) for m in d["messages"]) for d in train]
print(f"[ok] {src.name} -> data/mlx  train {len(train)} / valid {len(valid)} / test {len(test)}")
print(f"[check] train char-len max={max(lens)} mean={sum(lens)//len(lens)} (token≈char×1.3 for ko; v6 max_seq=3072 OK)")
