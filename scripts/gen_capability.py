"""v4 — generate gyaru-styled CAPABILITY Q&A (accurate content + full gyaru tone) so the
model answers technical questions in persona. LIGHT decoration keeps content readable.
Run: .venv/bin/python scripts/gen_capability.py --out data/_capability.jsonl"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import lexicon as lx
from decorator import Decorator

SYSTEM = lx.SYSTEM_PROMPT
_LATIN = re.compile(r"[A-Za-z]{4,}")      # drop long English leaks (short terms ok)
_HARMFUL = re.compile(r"자살|자해|살인|폭행")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/_capability.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=0)        # 0 = all questions
    a = ap.parse_args()
    import mlx.core as mx
    try: mx.set_cache_limit(3 * 1024 ** 3)
    except Exception: pass
    from stylizer import QwenMLXStylizer
    sty = QwenMLXStylizer(seed=a.seed)
    dec = Decorator(seed=a.seed + 2)
    qs = [l.strip() for l in (ROOT / "data/seed/questions_capability.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.n: qs = qs[:a.n]
    out = ROOT / a.out
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(qs):
            ans = sty.respond_capability(q)
            if not ans or len(ans) < 20 or _LATIN.search(ans) or _HARMFUL.search(ans):
                continue                                  # drop leaks / too-short / harmful
            ans = dec.gyaru(ans, profile="capability")     # 추임새 주입 O (가타카나 동일), 이모지/카오모지 절제
            fh.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans}]}, ensure_ascii=False) + "\n")
            n += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(qs)}] kept={n}")
            try: mx.clear_cache()
            except Exception: pass
    print(f"[done] {n} capability Q&A -> {out}")


if __name__ == "__main__":
    main()
