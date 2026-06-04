"""Phase 1b — yaho subsystem tests (scorer / extractor / renderer / pipeline)."""

import random

import pytest

from extractors import KiwiNounExtractor, RuleActivityExtractor
from renderers import ParaParaRenderer, YahoRenderer
from scorers import LexiconDepartureScorer, LexiconNegativityScorer


@pytest.fixture(scope="module")
def kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


# --- scorers --------------------------------------------------------------- #

def test_negativity_scorer_orders_by_intensity():
    s = LexiconNegativityScorer()
    strong = s.score("진짜 우울하다 시험 또 망쳤어")
    mild = s.score("발표 좀 걱정돼")
    neutral = s.score("오늘 날씨 좋고 기분 상쾌해")
    assert 0.0 <= neutral < mild < strong <= 1.0
    assert neutral == pytest.approx(0.0)


def test_departure_scorer_orders_by_intensity():
    s = LexiconDepartureScorer()
    leaving = s.score("나 이제 일하러 갈게 안녕")
    staying = s.score("오늘 점심 뭐 먹을지 고민이야")
    assert leaving > staying
    assert staying == pytest.approx(0.0)


# --- noun extractor (§5.3 / §16.6) ----------------------------------------- #

def test_extractor_prefers_proper_noun(kiwi):
    ex = KiwiNounExtractor(kiwi=kiwi)
    assert ex.extract("너 이러고 거제 가면 거제 시민들한테 혼나") == "거제"


def test_extractor_excludes_time_nouns(kiwi):
    ex = KiwiNounExtractor(kiwi=kiwi)
    assert ex.extract("오늘 점심 뭐 먹을지 고민이야") != "오늘"


def test_extractor_returns_none_when_no_noun(kiwi):
    ex = KiwiNounExtractor(kiwi=kiwi)
    assert ex.extract("응 그래 알았어 그렇구나") is None


def test_object_noun_extracted(kiwi):
    ex = KiwiNounExtractor(kiwi=kiwi)
    assert ex.extract("나 발표를 완전 망쳤어") == "발표"


# --- activity extractor (§5.4) --------------------------------------------- #

def test_activity_extractor_pattern():
    ex = RuleActivityExtractor()
    assert ex.extract("나 공부하러 도서관 간다") == "공부"
    assert ex.extract("밥 먹으러 가야겠다") in ("밥 먹", "먹")  # verb-phrase head
    assert ex.extract("그냥 심심해") is None


# --- renderers ------------------------------------------------------------- #

def test_yaho_renderer_format():
    r = YahoRenderer(rng=random.Random(0))
    out = r.render("거제")
    assert out.startswith("거제 야호~")


def test_parapara_renderer_template_and_generic():
    r = ParaParaRenderer(rng=random.Random(0))
    out = r.render("공부")
    assert "공부하는 동안" in out and "파라파라나 추고있어야겠다" in out and "💖" in out
    generic = r.render(None)
    assert generic.startswith("그럼 난 파라파라나 추고있어야겠다~") and "💖" in generic


# --- pipeline + factory ---------------------------------------------------- #

def test_yaho_pipeline_fires_strong_skips_neutral(kiwi):
    from pipeline_factory import build_yaho_pipeline
    pipe = build_yaho_pipeline(rng=random.Random(0), kiwi=kiwi)
    # strong negative, rng draw 0 -> always under firing prob -> fires
    fired = pipe.maybe_yaho("진짜 우울해 시험 또 망쳤어", rng=0.0)
    assert fired is not None and "야호~" in fired
    # neutral, high rng draw -> well above the ~p_base prob -> no fire
    assert pipe.maybe_yaho("오늘 날씨 좋고 기분 좋아", rng=0.99) is None


def test_factory_loads_gate_params_from_config():
    from pipeline_factory import build_yaho_pipeline
    pipe = build_yaho_pipeline(kiwi=None) if False else None  # avoid double Kiwi build
    from pipeline_factory import _load
    cfg = _load("configs/gate.yaml")["yaho"]
    assert (cfg["p_base"], cfg["p_max"], cfg["gamma"]) == (0.05, 0.55, 1.5)
