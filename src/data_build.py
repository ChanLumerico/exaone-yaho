"""§6 (adapted) — multi-turn gyaru corpus synthesis. OFFLINE teacher (§5.6).

PRIMARY seed: PersonaChat-Korean (MIT) — real multi-turn casual dialogues
(data/seed/personachat_ko.jsonl). We use the spec's stylization flow:
  1. map a dialogue's alternating utterances to user / assistant turns.
  2. for each ASSISTANT turn, arbiter (§5.5) decides on the PRECEDING user turn:
       PLAIN     -> Qwen stylizes the neutral turn to gyaru (meaning preserved, §16.4)
       YAHO      -> "{noun} 야호~" deflection replaces it (ignores content, by design)
       PARAPARA  -> stylized farewell + parapara line
  3. §5.2 light decoration (katakana + cleanup) on stylized turns.
  4. filter (style score, teacher-meme-leak guard, dedup) + per-sample JSONL log (§17).

This yields REAL multi-turn gyaru dialogues with persona held across turns. (SmileStyle
single-turn generation remains available via --source smilestyle.)

Run:  .venv/bin/python -m data_build --n 200 --out data/sft_multiturn.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import lexicon as lx  # noqa: E402
from decorator import Decorator  # noqa: E402
from gating import Policy  # noqa: E402
from logging_utils import JsonlLogger, new_run_id, write_manifest  # noqa: E402

SYSTEM = "너는 갸루 말투로 대답하는 AI야."
_DEFLECTION = re.compile(r"[가-힣A-Za-z0-9]+\s+야호")
_ALL_MARKERS = {t for e in lx.LEXICON.values() for (t, _o) in e} | \
               {o for e in lx.LEXICON.values() for (_t, o) in e}


def style_score(text: str) -> int:
    s = sum(m in text for m in _ALL_MARKERS)
    s += sum(e in text for e in lx.EMOJI_BANK) + sum(k in text for k in lx.KAOMOJI)
    s += any(t in text for t in ("~", "!!", "^^", "야호", "파라파라"))
    return s


def load_personachat(rng: random.Random, n: int, max_turns: int = 6) -> list[list[str]]:
    path = ROOT / "data/seed/personachat_ko.jsonl"
    dialogs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        turns = json.loads(line)["turns"]
        if len(turns) >= 2:
            dialogs.append([t for t in turns[:max_turns] if t.strip()])
    rng.shuffle(dialogs)
    return dialogs[:n]


def build_multiturn(turns, *, stylizer, arbiter, decorator, rng):
    """Build one multi-turn gyaru dialogue. Returns (messages|None, policy_counts)."""
    messages: list[dict] = []
    counts = {"yaho": 0, "parapara": 0, "plain": 0}
    for i, utt in enumerate(turns):
        if i % 2 == 0:
            messages.append({"role": "user", "content": utt})
            continue
        prev_user = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else None
        policy, meme = (Policy.PLAIN, None)
        if prev_user:
            policy, meme = arbiter.decide(prev_user, rng_yaho=rng.random(), rng_para=rng.random())
        if policy is Policy.YAHO and rng.random() < 0.35:
            text = meme                                   # punchy bare deflection
        else:
            # generation path with FULL running context -> coherent multi-turn gyaru
            resp = stylizer.respond(messages)
            if "파라파라" in resp or _DEFLECTION.search(resp):
                return None, counts                       # §16.4 teacher meme leak -> drop
            deco = decorator.light(resp)
            if policy is Policy.YAHO:                      # 손해-논리 등 나른 deflection + "{명사} 야호~"
                text = f"{deco} {meme}"
            elif policy is Policy.PARAPARA:
                text = f"{deco} {meme}"
            else:
                text = deco
        messages.append({"role": "assistant", "content": text})
        counts[policy.value] += 1
    if messages and messages[-1]["role"] == "user":
        messages.pop()                                    # end on an assistant turn
    return (messages if messages else None), counts


def synthesize(n=200, *, seed=42, model=None, max_turns=6, out="data/sft_multiturn.jsonl"):
    from pipeline_factory import build_arbiter
    from stylizer import QwenMLXStylizer

    rng = random.Random(seed)
    run_id = new_run_id("synth")
    log = JsonlLogger(phase="data", run_id=run_id, stage="synth")
    write_manifest(run_id, seed=seed, dataset_version="personachat-ko (MIT)",
                   base_model=model or "Qwen3-30B-A3B-Instruct-2507-4bit",
                   hardware="Apple M4 Max 36GB", extra={"phase": "1d"})

    sty_kw = {"seed": seed} if model is None else {"model_path": model, "seed": seed}
    stylizer = QwenMLXStylizer(**sty_kw)
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    arbiter = build_arbiter(rng=random.Random(seed + 1), kiwi=kiwi)
    decorator = Decorator(seed=seed + 2)

    dialogs = load_personachat(rng, n, max_turns=max_turns)
    out_path = ROOT / out if not Path(out).is_absolute() else Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    agg = {"accepted": 0, "rejected": 0, "yaho": 0, "parapara": 0, "plain": 0,
           "turns": 0, "multiturn": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, turns in enumerate(dialogs):
            messages, counts = build_multiturn(
                turns, stylizer=stylizer, arbiter=arbiter, decorator=decorator, rng=rng)
            if not messages:
                agg["rejected"] += 1
                log.sample(metrics={"accepted": False}, meta={"reason": "leak_or_empty"})
                continue
            a_turns = [m["content"] for m in messages if m["role"] == "assistant"]
            sc = min((style_score(t) for t in a_turns), default=0)
            key = " ".join(a_turns)[:80]
            if sc < 1 or key in seen:
                agg["rejected"] += 1
                log.sample(metrics={"accepted": False, "style_score": sc},
                           meta={"reason": "low_style" if sc < 1 else "dup"})
                continue
            seen.add(key)
            fh.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM}, *messages]},
                                ensure_ascii=False) + "\n")
            agg["accepted"] += 1
            agg["turns"] += len(a_turns)
            agg["multiturn"] += int(len(a_turns) >= 2)
            for k in ("yaho", "parapara", "plain"):
                agg[k] += counts[k]
            log.sample(metrics={"accepted": True, "style_score": sc, "n_assistant_turns": len(a_turns),
                                "yaho": counts["yaho"], "parapara": counts["parapara"], "plain": counts["plain"]})
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(dialogs)}] accepted={agg['accepted']} "
                      f"turns={agg['turns']} (y={agg['yaho']} p={agg['parapara']} plain={agg['plain']})")

    a = max(agg["accepted"], 1)
    tt = max(agg["turns"], 1)
    log.aggregate(n_dialogues=agg["accepted"], rejected=agg["rejected"],
                  multiturn_rate=agg["multiturn"] / a, avg_assistant_turns=agg["turns"] / a,
                  yaho_turn_rate=agg["yaho"] / tt, parapara_turn_rate=agg["parapara"] / tt,
                  plain_turn_rate=agg["plain"] / tt)
    print(f"\n[done] {agg['accepted']} dialogues ({agg['turns']} assistant turns) -> {out_path}")
    print(f"  multiturn={agg['multiturn']}/{agg['accepted']} | per-turn policy: "
          f"yaho={agg['yaho']} parapara={agg['parapara']} plain={agg['plain']} | rejected={agg['rejected']}")
    print(f"[ok] logged -> {log.path}")
    return out_path


def build_sft_corpus(config_path: str = "configs/style.yaml"):  # spec entry name
    return synthesize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="data/sft_multiturn.jsonl")
    a = ap.parse_args()
    synthesize(n=a.n, seed=a.seed, max_turns=a.max_turns, model=a.model, out=a.out)
