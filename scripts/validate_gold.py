"""Validate + spot-check the handcrafted gold dialogues (§6 / §16.4) and log
aggregate stats to JSONL (§17, phase=data).

Checks: JSONL parse, message structure & role alternation, yaho/parapara firing
distribution, §5.5 arbitration (no yaho+parapara in one response), marker/emoji
diversity (mode-collapse guard, §16.4), system-prompt consistency.

Run:  .venv/bin/python scripts/validate_gold.py [path]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# "{noun} 야호~" deflection = a word immediately before 야호 (NOT a sentence-initial
# greeting "야호~"). Used to distinguish the yaho POLICY from the greeting interjection.
_DEFLECTION = re.compile(r"[가-힣A-Za-z0-9]+\s+야호")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import lexicon as lx  # noqa: E402
from logging_utils import JsonlLogger, new_run_id  # noqa: E402

SYSTEM_EXPECTED = "너는 갸루 말투로 대답하는 AI야."
ALL_MARKERS = [t for entries in lx.LEXICON.values() for (t, _o) in entries]


def load(path: Path) -> list[dict]:
    rows, errors = [], []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append((i, str(e)))
    if errors:
        for ln, e in errors:
            print(f"  [PARSE ERROR] line {ln}: {e}")
    return rows


def check_structure(d: dict, idx: int) -> list[str]:
    issues = []
    msgs = d.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return [f"dialogue {idx}: missing/empty 'messages'"]
    roles = [m.get("role") for m in msgs]
    for j, m in enumerate(msgs):
        if m.get("role") not in {"system", "user", "assistant"}:
            issues.append(f"dialogue {idx} msg {j}: bad role {m.get('role')!r}")
        if not str(m.get("content", "")).strip():
            issues.append(f"dialogue {idx} msg {j}: empty content")
    # role order: optional system at 0, then alternate user/assistant, end assistant
    body = roles[1:] if roles and roles[0] == "system" else roles
    if "system" in body:
        issues.append(f"dialogue {idx}: 'system' not at index 0")
    expected = ["user", "assistant"]
    for k, r in enumerate(body):
        if r != expected[k % 2]:
            issues.append(f"dialogue {idx}: role order break at turn {k} (got {r})")
            break
    if body and body[-1] != "assistant":
        issues.append(f"dialogue {idx}: does not end with an assistant turn")
    return issues


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/gold/gold.jsonl"
    rows = load(path)
    print(f"\n=== gold validation: {path} ===")
    print(f"dialogues parsed: {len(rows)}")

    struct_issues, arb_violations = [], []
    n_assistant_turns = 0
    yaho_dlg = parapara_dlg = neither_dlg = 0
    turns_per = []
    marker_counter: Counter = Counter()
    emoji_counter: Counter = Counter()
    kaomoji_counter: Counter = Counter()
    systems: Counter = Counter()

    for i, d in enumerate(rows):
        struct_issues += check_structure(d, i)
        msgs = d.get("messages", [])
        a_turns = [m for m in msgs if m.get("role") == "assistant"]
        n_assistant_turns += len(a_turns)
        turns_per.append(len(a_turns))
        for m in msgs:
            if m.get("role") == "system":
                systems[m.get("content", "")] += 1
        dlg_has_yaho = dlg_has_para = False
        for m in a_turns:
            c = m.get("content", "")
            has_deflect = bool(_DEFLECTION.search(c))   # "{noun} 야호~" POLICY
            has_p = "파라파라" in c
            dlg_has_yaho |= has_deflect
            dlg_has_para |= has_p
            if has_deflect and has_p:
                arb_violations.append(i)  # §5.5 — both POLICIES in ONE response
            for mk in ALL_MARKERS:
                if mk in c:
                    marker_counter[mk] += 1
            for e in lx.EMOJI_BANK:
                if e in c:
                    emoji_counter[e] += 1
            for k in lx.KAOMOJI:
                if k in c:
                    kaomoji_counter[k] += 1
        if dlg_has_yaho:
            yaho_dlg += 1
        if dlg_has_para:
            parapara_dlg += 1
        if not dlg_has_yaho and not dlg_has_para:
            neither_dlg += 1

    n = max(len(rows), 1)
    print(f"assistant turns: {n_assistant_turns} | turns/dialogue: "
          f"min {min(turns_per)} / max {max(turns_per)} / avg {sum(turns_per)/n:.1f}")
    print(f"\n-- firing distribution (by dialogue) --")
    print(f"  yaho present:     {yaho_dlg:>3} ({yaho_dlg/n:.0%})")
    print(f"  parapara present: {parapara_dlg:>3} ({parapara_dlg/n:.0%})")
    print(f"  neither (neutral):{neither_dlg:>3} ({neither_dlg/n:.0%})")
    print(f"\n-- diversity (mode-collapse guard, §16.4) --")
    print(f"  distinct markers used: {len(marker_counter)}/{len(ALL_MARKERS)}")
    print(f"  top markers: {marker_counter.most_common(8)}")
    print(f"  distinct emojis: {len(emoji_counter)}/{len(lx.EMOJI_BANK)} | "
          f"distinct kaomoji: {len(kaomoji_counter)}/{len(lx.KAOMOJI)}")
    print(f"\n-- consistency --")
    print(f"  system prompts: {dict(systems)}")
    sys_ok = set(systems) == {SYSTEM_EXPECTED}

    print(f"\n-- correctness --")
    print(f"  structure issues: {len(struct_issues)}")
    for s in struct_issues[:15]:
        print(f"    - {s}")
    print(f"  §5.5 arbitration violations (yaho+parapara in one response): "
          f"{len(arb_violations)}" + (f" -> dialogues {arb_violations}" if arb_violations else ""))

    # §17 aggregate log (phase=data)
    log = JsonlLogger(phase="data", run_id=new_run_id("gold-validate"), stage="gold")
    log.aggregate(
        n_dialogues=len(rows), n_assistant_turns=n_assistant_turns,
        avg_turns=sum(turns_per) / n,
        yaho_rate=yaho_dlg / n, parapara_rate=parapara_dlg / n, neutral_rate=neither_dlg / n,
        distinct_markers=len(marker_counter), distinct_emojis=len(emoji_counter),
        distinct_kaomoji=len(kaomoji_counter),
        structure_issues=len(struct_issues), arbitration_violations=len(arb_violations),
        system_consistent=sys_ok,
    )
    print(f"\n[ok] aggregate logged -> {log.path}")

    verdict = (not struct_issues) and (not arb_violations) and sys_ok
    print(f"\n=== VERDICT: {'PASS' if verdict else 'NEEDS REVIEW'} ===")


if __name__ == "__main__":
    main()
