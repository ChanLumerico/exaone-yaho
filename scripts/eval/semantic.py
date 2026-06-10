"""Semantic eval (bge-m3): Acc_style (trained gyaru-vs-neutral classifier, primary) +
Sim (response<->prompt relevance cosine) + J = Acc_style*Sim. Writes logs/eval/sem_metrics.json."""
import sys, json, random, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng=random.Random(42)
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]

# ---- classifier training data: gyaru (corpus assistant turns) vs neutral (PersonaChat) ----
pos=[m["content"] for d in jl("data/gyaru_corpus_4.jsonl") for m in d["messages"] if m["role"]=="assistant"]
rng.shuffle(pos); pos=pos[:350]
neg=[]
for d in jl("data/seed/personachat_ko.jsonl"):
    neg+= [t for t in d["turns"] if len(t)>=8]
rng.shuffle(neg); neg=neg[:350]
print(f"classifier train: {len(pos)} gyaru / {len(neg)} neutral", flush=True)

print("loading bge-m3...", flush=True)
emb=SentenceTransformer("BAAI/bge-m3")
def E(texts): return emb.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=16)

X=np.vstack([E(pos),E(neg)]); y=np.array([1]*len(pos)+[0]*len(neg))
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
clf=LogisticRegression(max_iter=2000,C=1.0).fit(Xtr,ytr)
clf_acc=clf.score(Xte,yte)
print(f"Acc_style classifier held-out acc: {clf_acc:.3f}", flush=True)

tags=["base24","base78","v2","v3","v4","v5","v6","v8","v8b","orpo","orpo2","v9","v9orpo","v10","v10orpo","v11","v11orpo","v12","v12orpo","v121orpo","v122orpo"]
res={"_classifier_heldout_acc":clf_acc}
for t in tags:
    p=f"data/eval/gen_{t}.jsonl"
    if not Path(p).exists(): continue
    rows=jl(p); resp=[r["response"] for r in rows]; prompts=[r["prompt"] for r in rows]
    Re=E(resp); Pe=E(prompts)
    acc_style=float(clf.predict_proba(Re)[:,1].mean())             # mean P(gyaru)
    sim=float(np.mean([np.dot(Re[i],Pe[i]) for i in range(len(resp))]))  # relevance
    J=acc_style*sim
    res[t]=dict(acc_style=acc_style, sim=sim, J=J)
    print(f"  [{t}] Acc_style={acc_style:.3f} Sim={sim:.3f} J={J:.3f}", flush=True)
json.dump(res, open("logs/eval/sem_metrics.json","w"), ensure_ascii=False, indent=1)
print("-> logs/eval/sem_metrics.json")
