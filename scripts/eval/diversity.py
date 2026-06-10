"""Phase-0 diversity measurement suite (theory-grounded).
  * Vendi Score (Friedman & Dieng 2022): exp(Shannon entropy of normalized
    eigenvalues of the bge-m3 cosine kernel) = "effective # of distinct samples".
  * distinct-1..4 curve, Zipf coefficient, type-token ratio.
  * over-represented n-gram extraction (cross-response document-frequency) ->
    the data-driven stock-phrase BLOCKLIST that feeds Tier1 n-gram capping.
Run: .venv/bin/python scripts/eval/diversity.py
"""
import json, re
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
VERS = ["base78","v2","v3","v4","v5","v6","v8","v8b","orpo","orpo2","v9","v9orpo","v10","v10orpo","v11","v11orpo","v12","v12orpo","v121orpo","v122orpo"]
def load(v):
    p = ROOT/f"data/eval/gen_{v}.jsonl"
    return [json.loads(l)["response"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []

def vendi(X):                                   # X: (n,d) unit-norm embeddings
    n = len(X); K = (X @ X.T) / n               # trace(K)=1, eigvals sum to 1
    w = np.linalg.eigvalsh(K); w = w[w > 1e-12]; w = w/w.sum()
    return float(np.exp(-np.sum(w*np.log(w))))  # effective # distinct in [1,n]

def toks(t): return re.findall(r"[가-힣]+|[ぁ-んァ-ヶー一-龯]+|[A-Za-z]+|[~!?.,]", t)
def distinct_n(texts, n):
    g = Counter()
    for t in texts:
        x = toks(t)
        for i in range(len(x)-n+1): g[tuple(x[i:i+n])] += 1
    tot = sum(g.values()); return (len(g)/tot if tot else 0.0)
def zipf_coef(texts):
    c = Counter()
    for t in texts: c.update(toks(t))
    f = np.array(sorted(c.values(), reverse=True), dtype=float)
    if len(f) < 5: return 0.0
    r = np.arange(1, len(f)+1)
    return float(np.polyfit(np.log(r), np.log(f), 1)[0])
def ttr(texts):
    c = Counter()
    for t in texts: c.update(toks(t))
    tot = sum(c.values()); return (len(c)/tot if tot else 0.0)
def stock_phrases(texts, nmin=3, nmax=7, min_df=3):
    """n-grams by DOCUMENT frequency (# distinct responses containing them)."""
    out = []
    for n in range(nmin, nmax+1):
        df = Counter()
        for t in texts:
            x = toks(t); seen = set(tuple(x[i:i+n]) for i in range(len(x)-n+1))
            for g in seen: df[g] += 1
        for g, d in df.items():
            if d >= min_df: out.append((d, n, " ".join(g)))
    # keep longer phrases preferentially; sort by df then length
    out.sort(key=lambda z: (-z[0], -z[1]))
    return out

print("loading bge-m3...", flush=True)
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("BAAI/bge-m3")
res = {}
for v in VERS:
    txt = load(v)
    if not txt: continue
    X = emb.encode(txt, normalize_embeddings=True, show_progress_bar=False, batch_size=16)
    res[v] = {"n":len(txt), "vendi":round(vendi(np.asarray(X)),2),
              "vendi_norm":round(vendi(np.asarray(X))/len(txt),3),
              "d1":round(distinct_n(txt,1),3),"d2":round(distinct_n(txt,2),3),
              "d3":round(distinct_n(txt,3),3),"d4":round(distinct_n(txt,4),3),
              "zipf":round(zipf_coef(txt),3),"ttr":round(ttr(txt),3)}
(ROOT/"logs/eval/diversity_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{'ver':7} {'Vendi':>6} {'VS/n':>5} {'d1':>5} {'d2':>5} {'d3':>5} {'d4':>5} {'Zipf':>6} {'TTR':>5}")
for v,m in res.items():
    print(f"{v:7} {m['vendi']:>6} {m['vendi_norm']:>5} {m['d1']:>5} {m['d2']:>5} {m['d3']:>5} {m['d4']:>5} {m['zipf']:>6} {m['ttr']:>5}")

# v6 stock-phrase blocklist (the Tier1 input)
sp = stock_phrases(load("v6"))
blk = [{"df":d,"n":n,"phrase":p} for d,n,p in sp]
(ROOT/"logs/eval/v6_stock_phrases.json").write_text(json.dumps(blk, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n=== v6 과대표집 구문 (df>=3, n=3..7) — Tier1 블록리스트 후보 상위 18 ===")
for d,n,p in sp[:18]: print(f"  df={d:2}  {n}-gram  「{p}」")
print(f"\n[ok] -> logs/eval/diversity_metrics.json, v6_stock_phrases.json ({len(blk)} phrases)")
