"""Phase 1c DoD — parapara firing curve + §5.5 arbitration audit.

(1) departure firing rate follows the §5.4 gate curve; neutral stays low.
(2) over a mixed set (negative / departure / both / neutral), the arbiter NEVER
    emits yaho + parapara in one output (§5.5) — must be 0.
Logs curve points + the double-fire count to JSONL (§17).

Run:  .venv/bin/python scripts/validate_parapara_curve.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gating import GateParams, Policy, gate_prob  # noqa: E402
from logging_utils import JsonlLogger, new_run_id  # noqa: E402
from pipeline_factory import _load, build_arbiter, build_parapara_pipeline  # noqa: E402

_DEFLECTION = re.compile(r"[가-힣A-Za-z0-9]+\s+야호")

DEPARTURE = {
    "strong": [
        "나 이제 일하러 갈게 안녕", "졸려서 먼저 잘게 잘자", "회의 들어가야 해서 이만",
        "바빠서 이만 가볼게 나중에 얘기하자", "이제 진짜 갈게 안녕", "약속 있어서 먼저 나가볼게",
        "출근한다 다녀올게", "공부하러 도서관 갈게", "씻고 잘게 피곤하다", "심부름 다녀올게 금방 옴",
    ],
    "neutral": [
        "오늘 점심 뭐 먹을지 고민이야", "주말에 갈 카페 추천해줘", "요즘 노래 추천해줘",
        "강아지 키우고 싶어", "여행 어디가 좋을까", "옷 코디 추천", "오늘 날씨 좋다",
        "영화 추천해줘", "생일 선물 뭐 살까", "머리 스타일 바꿀까",
    ],
}
MIXED = [  # for the arbitration audit (incl. negative+departure)
    "짜증나서 그냥 집에 갈래", "시험 망쳐서 우울한데 이제 자러 갈게", "일하러 가야 되는데 가기 싫어 죽겠다",
    "면접 떨어졌어 진짜 최악", "공부하러 도서관 간다", "졸려서 먼저 잘게", "오늘 날씨 진짜 좋다",
    "발표 폭망했어 쪽팔려", "밥 먹으러 가야겠다", "남친이랑 싸워서 짜증나",
]
M = 3000


def main() -> None:
    rng = random.Random(0)
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    pipe = build_parapara_pipeline(rng=rng, kiwi=kiwi)
    arb = build_arbiter(rng=rng, kiwi=kiwi)
    p = _load("configs/gate.yaml")["parapara"]
    params = GateParams(p_base=p["p_base"], p_max=p["p_max"], gamma=p["gamma"])
    log = JsonlLogger(phase="data", run_id=new_run_id("parapara-curve"), stage="parapara")
    mc = random.Random(42)

    print("\n=== departure firing (parapara) ===")
    for cls in ("neutral", "strong"):
        rates, scores = [], []
        for q in DEPARTURE[cls]:
            s = pipe.scorer.score(q)
            fp = gate_prob(s, params)
            fires = sum(mc.random() < fp for _ in range(M)) / M
            scores.append(s)
            rates.append(fires)
        ms, mr = sum(scores) / len(scores), sum(rates) / len(rates)
        print(f"  {cls:8s}: mean s_dep={ms:.3f}  fire_rate={mr:.3f}")
        log.event("curve", metrics={"mean_s_dep": ms, "firing_rate": mr},
                  meta={"class": cls, "kind": "parapara"})

    print("\n=== §5.5 arbitration audit (mixed turns) ===")
    counts = {Policy.YAHO: 0, Policy.PARAPARA: 0, Policy.PLAIN: 0}
    double_fire = 0
    for q in MIXED:
        for _ in range(M // 10):
            policy, text = arb.decide(q, rng_yaho=mc.random(), rng_para=mc.random())
            counts[policy] += 1
            if text and _DEFLECTION.search(text) and "파라파라" in text:
                double_fire += 1
    total = sum(counts.values())
    print(f"  policies over {total} draws: yaho={counts[Policy.YAHO]} "
          f"parapara={counts[Policy.PARAPARA]} plain={counts[Policy.PLAIN]}")
    print(f"  yaho+parapara in ONE output: {double_fire}  (§5.5 requires 0)")
    log.event("arbitration_audit",
              metrics={"yaho": counts[Policy.YAHO], "parapara": counts[Policy.PARAPARA],
                       "plain": counts[Policy.PLAIN], "double_fire": double_fire})

    # DoD: departure fires more than neutral, and zero double-fire
    dep_mean = sum(pipe.scorer.score(q) for q in DEPARTURE["strong"]) / 10
    neu_mean = sum(pipe.scorer.score(q) for q in DEPARTURE["neutral"]) / 10
    ok = dep_mean > neu_mean and double_fire == 0
    print(f"\n=== VERDICT: {'PASS' if ok else 'REVIEW'} "
          f"(departure {dep_mean:.2f} > neutral {neu_mean:.2f}; double_fire={double_fire}) ===")
    print(f"[ok] logged -> {log.path}")


if __name__ == "__main__":
    main()
