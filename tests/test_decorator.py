"""Unit tests for the §5.2 decorator + postprocessor (src/decorator.py)."""

import random
import re

import lexicon as lx
import pytest
from decorator import DecoratorParams, Decorator, decorate, postprocess

JP = re.compile(r"[぀-ヿ一-鿿]")  # hiragana / katakana / CJK
MECH_CATS = ("interjection", "emphasis", "filler", "agreement")
MECH_MARKERS = {t for c in MECH_CATS for (t, _o) in lx.LEXICON[c]} | {
    o for c in MECH_CATS for (_t, o) in lx.LEXICON[c]
}


def mkparams(**over) -> DecoratorParams:
    base = dict(
        lambda_range=(0.2, 0.6), p_kata=0.12, p_tail_after_marker=0.7,
        emoji_boundary_prob_factor=0.5, ending_emoji_count=(1, 3),
        p_trailing_kaomoji=0.5, max_ending_per_clause=1, collapse_repeats=True,
        max_tail_run=2, mechanical_categories=MECH_CATS,
    )
    base.update(over)
    return DecoratorParams(**base)


def test_from_yaml_loads_real_config():
    p = DecoratorParams.from_yaml()
    assert p.lambda_range == (0.2, 0.6)
    assert p.p_kata == pytest.approx(0.12)
    assert set(p.mechanical_categories) == set(MECH_CATS)


def test_determinism_same_seed():
    p = mkparams()
    a = decorate("안녕 반가워 오늘 뭐해", 0.5, params=p, rng=random.Random(7))
    b = decorate("안녕 반가워 오늘 뭐해", 0.5, params=p, rng=random.Random(7))
    assert a == b


def test_lambda_zero_no_interjection_prepended():
    # lam=0 -> rule 1 never fires, no boundary emoji; core text is preserved up front
    out = decorate("안녕 반가워", 0.0, params=mkparams(), rng=random.Random(1))
    assert out.startswith("안녕 반가워")


def test_high_lambda_prepends_a_mechanical_marker():
    for seed in range(8):
        out = decorate("안녕 반가워", 1.0, params=mkparams(), rng=random.Random(seed))
        assert not out.startswith("안녕")  # something was prepended
        assert any(out.startswith(m) for m in MECH_MARKERS)


def test_katakana_substitution_when_p_kata_one():
    # lam=1 forces a prepend; p_kata=1 forces the ORIGINAL (Japanese) form
    out = decorate("안녕 반가워", 1.0, params=mkparams(p_kata=1.0), rng=random.Random(3))
    assert JP.search(out)


def test_only_mechanical_categories_injected():
    # context-only markers (e.g. 오시/누마/마타네/테헤페로) must never be prepended
    context_only = {t for c in ("affection", "fandom", "greeting", "reaction")
                    for (t, _o) in lx.LEXICON[c]}
    for seed in range(40):
        out = decorate("오늘 뭐 먹을까", 1.0, params=mkparams(p_kata=0.0), rng=random.Random(seed))
        first = out.split(" ")[0]
        assert not any(first.startswith(m) for m in context_only)


def test_postprocess_collapses_tail_runs():
    p = mkparams()
    assert postprocess("야호~~~~~", params=p) == "야호~~"
    assert postprocess("멋져!!!!", params=p) == "멋져!!"
    assert postprocess("좋아^^^^^", params=p) == "좋아^^"


def test_postprocess_collapses_duplicate_emoji():
    assert postprocess("좋아✨✨✨", params=mkparams()) == "좋아✨"


def test_postprocess_tidies_whitespace_and_punctuation():
    p = mkparams()
    assert postprocess("a    b", params=p) == "a b"
    assert postprocess("끝 !", params=p) == "끝!"


def test_decorate_output_is_clean_and_nonempty():
    p = mkparams()
    out = decorate("카페 추천해줘! 분위기 좋은 데로 가자", 0.6, params=p, rng=random.Random(11))
    assert out
    assert not re.search(r"[~!^]{3,}", out)            # no long tail runs
    for e in lx.EMOJI_BANK:
        assert (e + e) not in out                       # no duplicate-emoji runs
    assert "카페" in out and "가자" in out               # content survives


def test_decorator_class_samples_lambda_in_range():
    dec = Decorator(params=mkparams(), seed=0)
    for _ in range(20):
        assert 0.2 <= dec.sample_lambda() <= 0.6
    assert dec("안녕 오늘 날씨 좋다")  # callable end-to-end
