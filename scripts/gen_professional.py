"""v4 — generate PROFESSIONAL gyaru data: accurate content + gyaru tone. Uses
respond_capability (few-shot anchored, accurate) + decorator.gyaru(profile='capability')
(추임새 O, light extras) + NO memes/gating. Multi-turn (koVast) or single (handcrafted).
Run: .venv/bin/python scripts/gen_professional.py --source kovast --n 120 --out data/_professional.jsonl"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import lexicon as lx
from decorator import Decorator
_MW = re.compile(r"\s*마이웨이(?:로 말하면|로 생각하면|로 설명|로|는|야|지)?\s*")  # v4 — strip from professional

SYSTEM = lx.SYSTEM_PROMPT
_LATIN = re.compile(r"[A-Za-z]{4,}")
_HARMFUL = re.compile(r"자살|자해|살인|폭행")
_INSTR_LEAK = ("응답당", "추임새", "갸루어", "프롬프트", "리스트째", "손해 논리", "캐치프레이즈", "비유법")
def _bad(t: str) -> bool:
    return bool(_LATIN.search(t) or _HARMFUL.search(t) or any(w in t for w in _INSTR_LEAK))


def _seeds(source: str, in_path, n: int):
    if source == "kovast":
        rows = [json.loads(l)["user_turns"] for l in Path(in_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:  # handcrafted single-turn
        rows = [[q.strip()] for q in (ROOT / "data/seed/questions_capability.txt").read_text(encoding="utf-8").splitlines() if q.strip()]
    return rows[:n] if n else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kovast", "handcrafted"], default="kovast")
    ap.add_argument("--in", dest="in_path", default="data/seed/professional_kovast.jsonl")
    ap.add_argument("--out", default="data/_professional.jsonl")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import mlx.core as mx
    try: mx.set_cache_limit(3 * 1024 ** 3)
    except Exception: pass
    from stylizer import QwenMLXStylizer
    sty = QwenMLXStylizer(seed=a.seed)
    dec = Decorator(seed=a.seed + 2)
    seeds = _seeds(a.source, a.in_path, a.n)
    out = ROOT / a.out
    prog = out.with_suffix(out.suffix + ".progress")            # v4 — crash/interrupt resume
    done, mode = -1, "w"
    if prog.exists() and out.exists():
        try:
            done, mode = int(prog.read_text().strip() or "-1"), "a"
            print(f"  [resume] skipping 0..{done}")
        except Exception:
            done, mode = -1, "w"
    n_ok = n_turns = 0
    with out.open(mode, encoding="utf-8") as fh:
        for i, user_turns in enumerate(seeds):
            if i <= done:
                continue
            messages, ok = [], True
            for u in user_turns:
                messages.append({"role": "user", "content": u})
                resp = sty.respond_capability(messages)          # accurate + gyaru, few-shot anchored
                if not resp or len(resp) < 15 or _bad(resp):
                    ok = False; break
                messages.append({"role": "assistant", "content": _MW.sub(" ", dec.gyaru(resp, profile="capability")).strip()})
            if ok and len(messages) >= 2:
                fh.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM}, *messages]}, ensure_ascii=False) + "\n")
                fh.flush(); n_ok += 1; n_turns += sum(1 for m in messages if m["role"] == "assistant")
            prog.write_text(str(i))                              # advance checkpoint every seed
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(seeds)}] kept={n_ok} turns={n_turns}")
            try: mx.clear_cache()
            except Exception: pass
    print(f"[done] {n_ok} professional dialogues ({n_turns} assistant turns) -> {out}")


if __name__ == "__main__":
    main()
