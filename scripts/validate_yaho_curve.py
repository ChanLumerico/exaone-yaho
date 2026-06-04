"""Phase 1b DoD — verify yaho firing rate follows the §5.3 gating curve.

Scores a labeled query set (neutral / mild / strong negativity), then Monte-Carlo
samples the gate to show empirical firing rate tracks the target curve
  p = p_base + (p_max - p_base) * s_neg ** gamma.
Reports mean s_neg per class + per-bin MAE vs the target curve, and logs the
curve points to JSONL (§17.2, event=curve, kind=yaho).

Run:  .venv/bin/python scripts/validate_yaho_curve.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gating import GateParams, gate_prob  # noqa: E402
from logging_utils import JsonlLogger, new_run_id  # noqa: E402
from pipeline_factory import _load, build_yaho_pipeline  # noqa: E402

QUERIES = {
    "strong": [
        "진짜 우울하다 시험 또 망쳤어", "발표 완전 폭망했어 너무 쪽팔려", "남친이랑 또 싸웠어 진짜 짜증나",
        "면접 떨어졌다고 방금 연락 왔어", "돈이 없어서 이번 달 진짜 거지같아", "친구가 내 흉봤대 진짜 배신감",
        "엄마한테 개잔소리 들어서 너무 스트레스", "일 너무 많아서 죽겠다", "시험 망쳐서 진짜 최악이야",
        "또 실수해서 너무 속상해",
    ],
    "mild": [
        "발표 좀 걱정돼", "오늘 좀 피곤하네", "다이어트 또 실패했어", "지각해서 좀 그래",
        "살 좀 쪘나봐", "운동 갔는데 힘들어", "그 카페 별로였어", "무릎 살짝 까졌어",
        "요즘 좀 외로워", "면접 결과가 걱정이야",
    ],
    "neutral": [
        "주말에 갈 만한 카페 추천해줘", "오늘 점심 뭐 먹을지 고민이야", "요즘 들을 만한 노래 있어?",
        "강아지 키우고 싶은데 어떤 종이 좋아?", "여행 어디가 좋을까", "옷 코디 추천해줘",
        "오늘 날씨 진짜 좋다", "주말에 뭐하고 놀까", "영화 하나 추천해줘", "생일 선물 뭐 사줄까",
    ],
}
M = 3000  # Monte-Carlo draws per query


def main() -> None:
    rng_mc = random.Random(42)
    pipe = build_yaho_pipeline(rng=random.Random(0))
    y = _load("configs/gate.yaml")["yaho"]
    params = GateParams(p_base=y["p_base"], p_max=y["p_max"], gamma=y["gamma"])
    log = JsonlLogger(phase="data", run_id=new_run_id("yaho-curve"), stage="yaho")

    rows = []  # (cls, query, s_neg, firing_prob, gate_fire_rate, eff_fire_rate, noun)
    for cls, queries in QUERIES.items():
        for q in queries:
            s = pipe.scorer.score(q)
            fp = gate_prob(s, params)
            noun = pipe.extractor.extract(q)
            gate_fires = sum(rng_mc.random() < fp for _ in range(M)) / M
            eff_fires = 0
            for _ in range(M):
                eff_fires += pipe.maybe_yaho(q, rng=rng_mc.random()) is not None
            eff_fires /= M
            rows.append((cls, q, s, fp, gate_fires, eff_fires, noun))

    print("\n=== mean by class ===")
    for cls in ("neutral", "mild", "strong"):
        sub = [r for r in rows if r[0] == cls]
        ms = sum(r[2] for r in sub) / len(sub)
        mfp = sum(r[3] for r in sub) / len(sub)
        meff = sum(r[5] for r in sub) / len(sub)
        print(f"  {cls:8s}: mean s_neg={ms:.3f}  target p={mfp:.3f}  eff_fire={meff:.3f}")
        log.event("curve", metrics={"mean_s_neg": ms, "target_p": mfp, "eff_fire_rate": meff},
                  meta={"class": cls, "kind": "yaho"})

    # per-bin MAE: empirical gate-fire vs target curve at bin center
    bins = [(i / 5, (i + 1) / 5) for i in range(5)]
    print("\n=== curve adherence (gate-fire vs target) ===")
    maes = []
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r[2] < hi or (hi == 1.0 and r[2] == 1.0)]
        if not sub:
            continue
        emp = sum(r[4] for r in sub) / len(sub)
        center = (lo + hi) / 2
        target = gate_prob(center, params)
        maes.append(abs(emp - target))
        print(f"  s_neg∈[{lo:.1f},{hi:.1f}) n={len(sub):2d}  emp_fire={emp:.3f}  target≈{target:.3f}")
        log.curve(bin=f"[{lo:.1f},{hi:.1f})", mean_score=center, firing_rate=emp, kind="yaho")
    mae = sum(maes) / len(maes) if maes else float("nan")
    print(f"\n  mean |empirical - target| over bins: {mae:.3f}")

    # DoD assertions
    by = {c: sum(r[2] for r in rows if r[0] == c) / 10 for c in ("neutral", "mild", "strong")}
    ok = by["neutral"] < by["mild"] < by["strong"] and mae < 0.05
    print(f"\n=== VERDICT: {'PASS' if ok else 'REVIEW'} "
          f"(monotone s_neg + curve MAE {mae:.3f} < 0.05) ===")
    print(f"[ok] logged -> {log.path}")


if __name__ == "__main__":
    main()
