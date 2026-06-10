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

SYSTEM = lx.SYSTEM_PROMPT                                  # v3 — compressed persona anchor (train==infer)
_DEFLECTION = re.compile(r"[가-힣A-Za-z0-9]+\s+야호")
# v3 teacher-leak detectors (run on RAW teacher output, before clean decoration):
_LATIN_LEAK = re.compile(r"[A-Za-z]{3,}")                  # English/romaji words -> teacher failure
_KANA_LEAK = re.compile(r"[぀-ヿ]")                         # raw kana (teacher told 한글 음차 only)
_HARMFUL = re.compile(r"자살|자해|죽어\s?버려|목\s?매|손목\s?긋|뛰어내려|살인|때려\s?죽|폭행|강간")  # v4 harmless guard
_CRISIS = re.compile(r"자살|자해|죽고\s?싶|죽고싶|사라지고\s?싶|그만두고\s?싶|살기\s?싫|목\s?매|손목|뛰어내|번아웃|우울|희망이?\s?없|의미\s?없|다\s?포기|현타|무기력")  # v5 distress -> crisis circuit-breaker (no meme)
_HARM_ENCOURAGE = re.compile(r"죽어\s?버려|죽는\s?게\s?(?:나|낫)|목\s?매(?:달|어)|손목\s?(?:긋|그어)|뛰어내려|자살\s?해|자살\s?하는\s?게|자해\s?해|살\s?가치\s?없|없어져")  # v5 — block ENCOURAGEMENT only; allow 자살예방/상담 on crisis turns
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


def load_smilestyle(rng, max_turns: int = 6):
    """v4 — SmileStyle 'formal' column segmented into multi-turn casual dialogues
    (empty rows separate dialogues). Clean formal-register casual seed (236 dialogues)."""
    import csv
    path = ROOT / "data/seed/smilestyle_dataset.tsv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header, rows = rows[0], rows[1:]
    fi = header.index("formal")
    segs, cur = [], []
    for r in rows:
        v = r[fi].strip() if len(r) > fi else ""
        if v:
            cur.append(v)
        elif cur:
            segs.append(cur[:max_turns]); cur = []
    if cur:
        segs.append(cur[:max_turns])
    rng.shuffle(segs)
    return [s for s in segs if len(s) >= 2]


def build_multiturn(turns, *, stylizer, arbiter, decorator, rng, neg_set=frozenset(), dep_set=frozenset(), gen_followups=0):
    """Build one multi-turn gyaru dialogue. Returns (messages|None, policy_counts)."""
    messages: list[dict] = []
    counts = {"yaho": 0, "parapara": 0, "plain": 0}
    for i, utt in enumerate(turns):
        if i % 2 == 0:
            messages.append({"role": "user", "content": utt})
            continue
        prev_user = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else None
        is_crisis = bool(prev_user and _CRISIS.search(prev_user))   # v5 — distress: suppress meme, teacher does crisis-breaker
        policy, meme = (Policy.PLAIN, None)
        if prev_user and not is_crisis:
            # v3 — curated triggers are *definitely* neg/departure; force the meme 80%
            # of the time (the lexicon scorer undercounts some inflections).
            ry = 0.0 if (prev_user in neg_set and rng.random() < 0.8) else rng.random()
            rp = 0.0 if (prev_user in dep_set and rng.random() < 0.8) else rng.random()
            policy, meme = arbiter.decide(prev_user, rng_yaho=ry, rng_para=rp)
        if policy is Policy.YAHO and rng.random() < 0.25:   # v4 — fewer bare (more 손해 substance)
            # v3 — punchy FRONT-loaded deflection (soulless reaction + "{noun} 야호~"),
            # so yaho is not learned only as a droppable terminal suffix.
            text = f"{rng.choice(lx.SOULLESS_REACTIONS)} {meme}"
        else:
            # generation path with FULL running context -> coherent multi-turn gyaru
            resp = stylizer.respond(messages)
            if ("파라파라" in resp or _DEFLECTION.search(resp)
                    or _LATIN_LEAK.search(resp) or _KANA_LEAK.search(resp) or (_HARM_ENCOURAGE if is_crisis else _HARMFUL).search(resp)
                    or any(w in resp for w in ("응답당", "추임새", "갸루어", "음차", "프롬프트", "리스트째", "손해 논리", "캐치프레이즈"))):
                return None, counts                       # teacher meme/instruction/code-switch/harmful leak -> drop
            deco = resp if is_crisis else decorator.gyaru(resp)   # v5 — crisis: minimal decoration (serious/sincere tone)
            if policy is Policy.YAHO:                      # 손해-논리 등 나른 deflection + "{명사} 야호~"
                text = f"{deco} {meme}"
            elif policy is Policy.PARAPARA:
                text = f"{deco} {meme}"
            else:
                text = deco
        messages.append({"role": "assistant", "content": text})
        counts[policy.value] += 1
    # v4 — coherent multiturn: extend with CONTEXTUALLY-generated follow-up user turns
    # (fixes incoherent random splices that taught the model to ignore prior context).
    for _ in range(gen_followups):
        if not (messages and messages[-1]["role"] == "assistant"):
            break
        nxt = (stylizer.next_user_turn(messages) or "").strip()
        if len(nxt) < 2 or _HARMFUL.search(nxt):
            break
        messages.append({"role": "user", "content": nxt})
        resp = stylizer.respond(messages)
        fu_crisis = bool(_CRISIS.search(messages[-1]["content"]))   # v5
        if ("파라파라" in resp or _DEFLECTION.search(resp) or _LATIN_LEAK.search(resp)
                or _KANA_LEAK.search(resp) or (_HARM_ENCOURAGE if fu_crisis else _HARMFUL).search(resp)):
            messages.pop()                                # drop dangling user turn
            break
        messages.append({"role": "assistant", "content": decorator.gyaru(resp)})
        counts["plain"] += 1
    if messages and messages[-1]["role"] == "user":
        messages.pop()                                    # end on an assistant turn
    return (messages if messages else None), counts


def _read_lines(rel):
    pth = ROOT / rel
    return [l.strip() for l in pth.read_text(encoding="utf-8").splitlines() if l.strip()] if pth.exists() else []


def _interleave(users):
    out = []
    for u in users:
        out += [u, ""]                                   # even=user turn, odd=ignored placeholder
    return out


def plan_dialogs(rng, n, max_turns=6):
    """v4 — stratified + COHERENT multiturn. Returns (turns, n_followups). ~1/3
    negativity-led (yaho) EXTENDED with contextually-generated follow-ups; ~1/3
    departure-closed (parapara) built on a REAL PersonaChat opening; ~1/3 neutral."""
    pc = load_personachat(rng, n * 2, max_turns=max_turns) + load_smilestyle(rng, max_turns=max_turns)
    rng.shuffle(pc)                                               # PersonaChat + SmileStyle casual pool
    neg = _read_lines("data/seed/triggers_neg.txt"); rng.shuffle(neg)
    dep = _read_lines("data/seed/triggers_dep.txt"); rng.shuffle(dep)
    third = max(n // 3, 1)
    plan, pci = [], 0
    for i in range(third):                                        # neg-led: trigger + 1-2 coherent followups
        nf = 2 if rng.random() < 0.7 else 1
        plan.append((_interleave([neg[i % len(neg)]]), nf))
    for i in range(third):                                        # dep-led: real coherent opening + dep last
        d = pc[pci % len(pc)]; pci += 1
        users = [u for u in d[::2][:2] if u.strip()]              # consecutive user turns (coherent)
        users.append(dep[i % len(dep)] if dep else "이만 가볼게")
        plan.append((_interleave(users), 0))
    food = _read_lines("data/seed/triggers_food.txt"); rng.shuffle(food)
    for i in range(max(n // 10, 1)):                             # 설렁탕 반전 (~10%, teacher does 3-step)
        plan.append((_interleave([food[i % len(food)]]) if food else pc[pci % len(pc)],
                     1 if rng.random() < 0.5 else 0))
    while len(plan) < n and pci < len(pc):                        # plain: coherent pc dialog
        plan.append((pc[pci], 0)); pci += 1
    rng.shuffle(plan)
    return plan[:n]


def synthesize(n=200, *, seed=42, model=None, max_turns=6, out="data/sft_multiturn.jsonl"):
    from pipeline_factory import build_arbiter
    from stylizer import QwenMLXStylizer

    rng = random.Random(seed)
    try:                                                  # v3 — cap Metal buffer pool (OOM headroom)
        import mlx.core as mx; mx.set_cache_limit(3 * 1024 ** 3)
    except Exception:
        pass
    run_id = new_run_id("synth")
    log = JsonlLogger(phase="data", run_id=run_id, stage="synth")
    write_manifest(run_id, seed=seed, dataset_version="personachat-ko (MIT)",
                   base_model=model or "Qwen3-30B-A3B-Instruct-2507-6bit-DWQ",
                   hardware="Apple M4 Max 36GB", extra={"phase": "1d"})

    sty_kw = {"seed": seed} if model is None else {"model_path": model, "seed": seed}
    stylizer = QwenMLXStylizer(**sty_kw)
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    arbiter = build_arbiter(rng=random.Random(seed + 1), kiwi=kiwi)
    decorator = Decorator(seed=seed + 2)

    dialogs = plan_dialogs(rng, n, max_turns=max_turns)
    neg_set = frozenset(_read_lines("data/seed/triggers_neg.txt"))
    dep_set = frozenset(_read_lines("data/seed/triggers_dep.txt"))
    out_path = ROOT / out if not Path(out).is_absolute() else Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    agg = {"accepted": 0, "rejected": 0, "yaho": 0, "parapara": 0, "plain": 0,
           "turns": 0, "multiturn": 0}
    progress_path = out_path.with_suffix(out_path.suffix + ".progress")
    done, mode = -1, "w"
    if progress_path.exists() and out_path.exists():
        try:
            done, mode = int(progress_path.read_text().strip() or "-1"), "a"
            print(f"  [resume] skipping dialogues 0..{done}")
        except Exception:
            done, mode = -1, "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        for idx, (turns, nf) in enumerate(dialogs):
            if idx <= done:
                continue
            messages, counts = build_multiturn(
                turns, stylizer=stylizer, arbiter=arbiter, decorator=decorator, rng=rng,
                neg_set=neg_set, dep_set=dep_set, gen_followups=nf)
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
            fh.flush(); progress_path.write_text(str(idx))    # v3 — crash-resume checkpoint
            agg["accepted"] += 1
            agg["turns"] += len(a_turns)
            agg["multiturn"] += int(len(a_turns) >= 2)
            for k in ("yaho", "parapara", "plain"):
                agg[k] += counts[k]
            log.sample(metrics={"accepted": True, "style_score": sc, "n_assistant_turns": len(a_turns),
                                "yaho": counts["yaho"], "parapara": counts["parapara"], "plain": counts["plain"]})
            if (idx + 1) % 5 == 0:                            # 6bit 메모리 누적 방지
                try:
                    import mlx.core as mx; mx.clear_cache()
                except Exception:
                    pass
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
