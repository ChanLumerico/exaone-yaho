"""Unit tests for §5.3/§5.4/§5.5 gating math (src/gating.py)."""

import pytest

from gating import GateParams, Policy, arbitrate, gate_prob

YAHO = GateParams(p_base=0.05, p_max=0.55, gamma=1.5)
PARA = GateParams(p_base=0.04, p_max=0.60, gamma=1.4)


def test_gate_endpoints():
    assert gate_prob(0.0, YAHO) == pytest.approx(0.05)
    assert gate_prob(1.0, YAHO) == pytest.approx(0.55)


def test_gate_clamps_out_of_range():
    assert gate_prob(-5.0, YAHO) == pytest.approx(0.05)
    assert gate_prob(2.0, YAHO) == pytest.approx(0.55)


def test_gate_monotonic_increasing():
    xs = [i / 20 for i in range(21)]
    ys = [gate_prob(x, YAHO) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))


def test_gate_convex_suppresses_weak_signal():
    # gamma > 1 => at the midpoint the prob is below the linear interpolant.
    linear_mid = (YAHO.p_base + YAHO.p_max) / 2
    assert gate_prob(0.5, YAHO) < linear_mid


def test_gateparams_validation():
    with pytest.raises(ValueError):
        GateParams(p_base=0.6, p_max=0.5, gamma=1.5)  # p_base > p_max
    with pytest.raises(ValueError):
        GateParams(p_base=0.1, p_max=0.5, gamma=0)  # gamma <= 0


def _arb(s_neg, s_dep, ry, rp, thresh=0.5):
    return arbitrate(
        s_neg=s_neg, s_dep=s_dep, yaho_params=YAHO, parapara_params=PARA,
        departure_priority_threshold=thresh, rng_yaho=ry, rng_para=rp,
    )


def test_arbitration_strong_departure_prefers_parapara():
    # both would fire, departure strong -> parapara wins (§5.5 rule 2)
    assert _arb(s_neg=0.9, s_dep=0.9, ry=0.0, rp=0.0) is Policy.PARAPARA


def test_arbitration_negativity_only():
    assert _arb(s_neg=0.9, s_dep=0.0, ry=0.0, rp=0.99) is Policy.YAHO


def test_arbitration_neither_fires_is_plain():
    # both draws above their (low, weak-signal) probabilities
    assert _arb(s_neg=0.1, s_dep=0.1, ry=0.99, rp=0.99) is Policy.PLAIN


def test_arbitration_weak_departure_higher_prob_wins():
    # departure below threshold -> compare probs; here yaho prob > para prob
    res = _arb(s_neg=0.95, s_dep=0.2, ry=0.0, rp=0.0, thresh=0.5)
    assert res is Policy.YAHO
