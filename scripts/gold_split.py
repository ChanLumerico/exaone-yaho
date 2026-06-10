"""Split gold by §16.4 role: meme-free (①) vs meme (yaho/parapara).

§16.4 role separation:
  * meme-free ① dialogues  -> Qwen bulk-stylizer few-shot (ordinary gyaru tone only)
  * meme dialogues (yaho/parapara) -> eval anchors + renderer reference + DPO chosen

A dialogue is "meme" if ANY assistant turn contains a "{noun} 야호~" deflection OR a
파라파라 line. Greeting "야호~" (sentence-initial) does NOT count (matches §5.3 policy
vs. interjection distinction used in validate_gold.py).

Writes data/gold/gold_plain.jsonl (①) and data/gold/gold_meme.jsonl.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/gold/gold.jsonl"
_DEFLECTION = re.compile(r"[가-힣A-Za-z0-9]+\s+야호")


def is_meme(dialogue: dict) -> bool:
    for m in dialogue["messages"]:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        if _DEFLECTION.search(c) or "파라파라" in c:
            return True
    return False


def main() -> None:
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    meme = [r for r in rows if is_meme(r)]
    plain = [r for r in rows if not is_meme(r)]

    (ROOT / "data/gold/gold_meme.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in meme) + "\n", encoding="utf-8"
    )
    (ROOT / "data/gold/gold_plain.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in plain) + "\n", encoding="utf-8"
    )
    print(f"[ok] total {len(rows)} | meme (yaho/parapara) {len(meme)} -> gold_meme.jsonl"
          f" | meme-free ① {len(plain)} -> gold_plain.jsonl")


if __name__ == "__main__":
    main()
