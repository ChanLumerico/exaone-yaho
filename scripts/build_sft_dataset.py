"""Phase 2 prep — combine synth corpus + gold into mlx-lm SFT data (§7.2).

Merges PersonaChat-synth (data/_synth_pc.jsonl) + gold (data/gold/gold.jsonl),
strips 토리마/とりま everywhere (user directive — keep it out of the whole corpus),
shuffles (seeded), splits train/valid/test, and writes data/mlx/{train,valid,test}.jsonl
in the chat format mlx_lm.lora expects. Prints a recommended `iters` for sft.yaml.

(Baseline run: gyaru corpus only — capability-replay (§7.4b) is the next iteration
once we measure the capability drop in Phase 3.)

Run:  .venv/bin/python scripts/build_sft_dataset.py [--synth data/_synth_pc.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from logging_utils import JsonlLogger, new_run_id  # noqa: E402
import lexicon as lx  # noqa: E402

_TORIMA = re.compile(r"\s*(?:토리마|とりま)[~,!?]*\s*")
_HRIM = re.compile(r"\s*흐림\s*처리(?:\s*해버려요?|\s*해버려|\s*하고|\s*해줘|\s*해|\s*함)?[~,.!?]*\s*")  # v4 — remove entirely (user)
_BANNED = re.compile(r"\s*(?:바빗타|バビった|루피)[~,.!?]*\s*")  # v4 — remove (user); 루피 too(뜻 없이 남발)
_SONHAE = re.compile(r"\s*[^~。.!?\n]*손해(?:예요|이에요|에요|야|이야|이에|니까|인데|잖아|지|래|임|네|라니까)?(?=[~,.!?\s]|$)[~,.!?]*")  # v5 — strip 손해-묵살 clause(서술형 포함; 손해도/손해는/배상 보호)

# v4 — keep ONLY gyaru emojis (EMOJI_BANK ∪ SPECIAL_CHARS ∪ KAOMOJI). Protect the
# allowed set (longest-first, covers ZWJ/VS sequences), strip all other emoji-range
# chars, restore. Guarantees gyaru-emoji-only regardless of what the teacher emitted.
_ALLOWED_VIS = sorted(set(lx.EMOJI_BANK) | set(lx.SPECIAL_CHARS) | set(lx.KAOMOJI), key=len, reverse=True)
_EMOJI_RANGE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF"
    "\U00002300-\U000023FF\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F\U0000200D\U00002190-\U000021FF]")

def strip_nongyaru_emoji(text: str) -> str:
    holders = {}
    for i, e in enumerate(_ALLOWED_VIS):
        ph = "\x00%d\x00" % i
        if e in text:
            text = text.replace(e, ph); holders[ph] = e
    text = _EMOJI_RANGE.sub("", text)
    for ph, e in holders.items():
        text = text.replace(ph, e)
    return text

# v3 — opener diversifier: the teacher opens ~91% of turns with "에~". Train on that
# and the model robotically starts every reply with "에~". Replace the leading
# reaction ~65% of the time with a varied opener (or none) so the corpus teaches
# diverse sentence openings. Applied uniformly to synth + gold at build time.
_OPENER_RE = re.compile(r"^(에+~+\??|에 )\s*")
_OPENERS = ["으음~", "우와아…", "오~", "음~", "헐~", "아~", "에이~", "오우~", "흐음~",
            "그치~", "오호~", "에~", "에~?", "", "", "", ""]

def _diversify_opener(text: str, rng) -> str:
    m = _OPENER_RE.match(text)
    if m and rng.random() < 0.65:
        rep = rng.choice(_OPENERS)
        rest = text[m.end():]
        return (rep + " " + rest).strip() if rep else rest.strip()
    return text


def _strip(d: dict) -> dict:
    for m in d["messages"]:
        if m["role"] == "assistant":
            c = strip_nongyaru_emoji(_SONHAE.sub(" ", _BANNED.sub(" ", _HRIM.sub(" ", _TORIMA.sub(" ", m["content"])))))   # v5 — +손해 strip; gyaru emojis only
            m["content"] = re.sub(r"\s{2,}", " ", c).strip()
    return d


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def prefix_expand(d: dict) -> list[dict]:
    """v3 — split a multi-turn dialogue into ONE example per assistant turn, each
    ending at that turn with full prior context, and force the fixed persona anchor
    as the system message. Fixes mlx-lm mask_prompt training only the LAST turn."""
    body = [m for m in d["messages"] if m["role"] != "system"]
    sys_msg = {"role": "system", "content": lx.SYSTEM_PROMPT}
    out = []
    for i, m in enumerate(body):
        if m["role"] == "assistant" and m["content"].strip():
            out.append({"messages": [sys_msg] + body[: i + 1]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/synth_v4.jsonl")
    ap.add_argument("--gold", default="data/gold/gold.jsonl")
    ap.add_argument("--professional", default="data/_professional.jsonl")   # v4 koVast multi-turn
    ap.add_argument("--capability", default="data/_capability.jsonl")       # v4 handcrafted single-turn
    ap.add_argument("--out", default="data/mlx")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--version", default="4")   # 종합 코퍼스 버전 -> data/gyaru_corpus_{v}.jsonl
    a = ap.parse_args()

    rng = random.Random(a.seed)
    synth = [_strip(d) for d in load(ROOT / a.synth)]
    gold = [_strip(d) for d in load(ROOT / a.gold)]
    prof = [_strip(d) for d in load(ROOT / a.professional)]   # koVast professional multi-turn
    cap = [_strip(d) for d in load(ROOT / a.capability)]      # handcrafted capability single-turn
    everyday = len(synth) + len(gold)
    professional = len(prof) + len(cap)
    print(f"[mix] everyday {everyday} (synth {len(synth)} + gold {len(gold)}) : "
          f"professional {professional} (koVast {len(prof)} + handcrafted {len(cap)})  "
          f"= {100*professional//max(everyday+professional,1)}% professional")
    data = synth + gold + prof + cap
    for d in data:                                   # v3 — break the "에~" opener monoculture
        for m in d["messages"]:
            if m["role"] == "assistant":
                m["content"] = _diversify_opener(m["content"], rng)
    rng.shuffle(data)
    if not data:
        raise SystemExit("no data — generate the corpus first")
    # 종합 갸루 코퍼스(병합본, 정제 완료)를 고정 이름으로 저장
    corpus_path = ROOT / f"data/gyaru_corpus_{a.version}.jsonl"
    corpus_path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in data) + "\n", encoding="utf-8")
    print(f"[ok] 종합 갸루 코퍼스 -> {corpus_path.name} ({len(data)} 대화)")

    n = len(data)
    n_val = max(8, n // 20)
    n_test = max(8, n // 20)
    valid, test, train = data[:n_val], data[n_val:n_val + n_test], data[n_val + n_test:]
    d_train, d_valid, d_test = len(train), len(valid), len(test)

    # v3 — prefix-expand AFTER the dialogue-level split (no prefix leaks across splits)
    def expand_split(split):
        seen, res = set(), []
        for d in split:
            for ex in prefix_expand(d):
                key = json.dumps(ex, ensure_ascii=False)
                if key not in seen:
                    seen.add(key); res.append(ex)
        return res
    valid, test, train = expand_split(valid), expand_split(test), expand_split(train)
    rng.shuffle(train)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        (out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in split) + "\n", encoding="utf-8")

    iters = max(100, (len(train) // a.batch_size) * a.epochs)
    leak = sum("토리마" in m["content"] or "とりま" in m["content"]
               for d in data for m in d["messages"] if m["role"] == "assistant")
    print(f"[ok] synth={len(synth)} + gold={len(gold)} = {n} dialogues "
          f"(split {d_train}/{d_valid}/{d_test}) -> prefix-expanded "
          f"train {len(train)} / valid {len(valid)} / test {len(test)} @ {out}")
    print(f"[ok] 토리마 residual: {leak} (목표 0)")
    print(f"[recommend] sft.yaml  iters: {iters}   (≈{a.epochs} epochs, batch {a.batch_size})")

    log = JsonlLogger(phase="data", run_id=new_run_id("sftset"), stage="build")
    log.aggregate(n_total=n, n_synth=len(synth), n_gold=len(gold),
                  train=len(train), valid=len(valid), test=len(test),
                  recommended_iters=iters, torima_residual=leak)


if __name__ == "__main__":
    main()
