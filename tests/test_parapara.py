"""Phase 1c — parapara pipeline + §5.5 arbitration tests."""

import random
import re

import pytest

from gating import Policy

_DEFLECTION = re.compile(r"[가-힣A-Za-z0-9]+\s+야호")


@pytest.fixture(scope="module")
def kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


def test_parapara_fires_departure_skips_neutral(kiwi):
    from pipeline_factory import build_parapara_pipeline
    pipe = build_parapara_pipeline(rng=random.Random(0), kiwi=kiwi)
    fired = pipe.maybe_parapara("나 이제 일하러 갈게 안녕", rng=0.0)
    assert fired is not None and "파라파라" in fired
    assert pipe.maybe_parapara("오늘 점심 뭐 먹을지 고민이야", rng=0.99) is None


def test_arbiter_departure_only_parapara(kiwi):
    from pipeline_factory import build_arbiter
    arb = build_arbiter(rng=random.Random(0), kiwi=kiwi)
    policy, text = arb.decide("나 이제 자러 갈게 잘자", rng_yaho=0.0, rng_para=0.0)
    assert policy is Policy.PARAPARA and "파라파라" in text


def test_arbiter_negative_only_yaho(kiwi):
    from pipeline_factory import build_arbiter
    arb = build_arbiter(rng=random.Random(0), kiwi=kiwi)
    policy, text = arb.decide("시험 또 망쳤어 진짜 우울해", rng_yaho=0.0, rng_para=0.99)
    assert policy is Policy.YAHO and "야호~" in text


def test_arbiter_strong_departure_wins_over_negative(kiwi):
    # both signals present, departure strong -> parapara wins (§5.5 rule 2)
    from pipeline_factory import build_arbiter
    arb = build_arbiter(rng=random.Random(0), kiwi=kiwi)
    policy, _ = arb.decide("진짜 짜증나서 이제 그만 갈게 안녕", rng_yaho=0.0, rng_para=0.0)
    assert policy is Policy.PARAPARA


def test_arbiter_neutral_is_plain(kiwi):
    from pipeline_factory import build_arbiter
    arb = build_arbiter(rng=random.Random(0), kiwi=kiwi)
    policy, text = arb.decide("오늘 날씨 진짜 좋고 기분 좋아", rng_yaho=0.99, rng_para=0.99)
    assert policy is Policy.PLAIN and text is None


def test_arbiter_never_emits_both_memes(kiwi):
    # §5.5 — across many turns/draws, a single output never has yaho AND parapara
    from pipeline_factory import build_arbiter
    arb = build_arbiter(rng=random.Random(1), kiwi=kiwi)
    turns = [
        "짜증나서 그냥 집에 갈래", "시험 망쳐서 우울한데 이제 자러 갈게",
        "일하러 가야 되는데 가기 싫어 죽겠다", "면접 떨어졌어", "공부하러 도서관 간다",
        "오늘 날씨 좋다",
    ]
    rng = random.Random(7)
    for _ in range(300):
        t = rng.choice(turns)
        _, text = arb.decide(t, rng_yaho=rng.random(), rng_para=rng.random())
        if text is not None:
            assert not (_DEFLECTION.search(text) and "파라파라" in text)
