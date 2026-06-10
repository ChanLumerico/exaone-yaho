"""v4 — load koVast professional MULTI-TURN user turns (the second seed). We consume
ONLY user turns (the Qwen teacher generates accurate-gyaru answers via respond_capability).
Output: normalized {source, user_turns:[...]} jsonl, Korean-only filtered.
Run: .venv/bin/python scripts/load_professional.py --n 150 --out data/seed/professional_kovast.jsonl"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

_LATIN = re.compile(r"[A-Za-z]{4,}")          # drop user turns with long English (rare in koVast)
_CONT = re.compile(r"^(그럼|그러면|그건|그래서|그러니까|그리고|근데|그래도|이건|저건|위에|아까|방금|앞서|이러한|그런)")  # orphaned continuation start
_SHARDS = [
    "data/train-00000-of-00003-e6c7f144913cac49.parquet",
    "data/train-00001-of-00003-76f1a8992c6a4ebe.parquet",
    "data/train-00002-of-00003-2e68fdb5fae74a85.parquet",
]


def load_kovast(n: int, *, max_user_turns: int = 4, min_user_turns: int = 2, take_every: int = 7):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    out, seen = [], 0
    for shard in _SHARDS:
        fp = hf_hub_download("maywell/koVast", shard, repo_type="dataset")
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=1000):
            for r in batch.to_pylist():
                seen += 1
                if seen % take_every:                       # subsample for topic diversity
                    continue
                conv = r.get("conversations") or []
                users = [c.get("value", "").strip() for c in conv
                         if c.get("from") == "user" and c.get("value", "").strip()]
                users = [u for u in users[:max_user_turns] if not _LATIN.search(u)]
                if len(users) >= min_user_turns and not _CONT.match(users[0]):   # self-contained start
                    out.append({"source": "kovast", "user_turns": users})
                    if len(out) >= n:
                        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="data/seed/professional_kovast.jsonl")
    a = ap.parse_args()
    rows = load_kovast(a.n)
    out = ROOT / a.out
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    nt = sum(len(r["user_turns"]) for r in rows)
    print(f"[ok] {len(rows)} professional dialogues ({nt} user turns, avg {nt/max(len(rows),1):.1f}) -> {out}")


if __name__ == "__main__":
    main()
