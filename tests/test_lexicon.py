"""Sanity tests for the §5.1 lexicon data (src/lexicon.py)."""

import lexicon as lx


def test_all_lexicon_entries_are_translit_original_pairs():
    for category, entries in lx.LEXICON.items():
        assert entries, f"{category} is empty"
        for pair in entries:
            assert len(pair) == 2, f"{category}: {pair} is not a 2-tuple"
            translit, original = pair
            assert translit and original


def test_mechanical_and_context_categories_partition_known_keys():
    mech = set(lx.MECHANICAL_CATEGORIES)
    ctx = set(lx.CONTEXT_ONLY_CATEGORIES)
    assert mech.isdisjoint(ctx)  # a category is mechanical XOR context-only
    assert (mech | ctx) <= set(lx.LEXICON)  # all reference real categories


def test_signature_meme_templates_present():
    assert "{noun}" in lx.YAHO_TEMPLATE
    assert "{activity}" in lx.PARAPARA_TEMPLATE
    assert lx.PARAPARA_GENERIC  # generic fallback exists


def test_banks_nonempty():
    assert lx.EMOJI_BANK and lx.KAOMOJI and lx.TAILS
