"""Unit tests for the §17 JSONL logging skeleton (src/logging_utils.py)."""

import importlib
import json

import pytest


@pytest.fixture()
def logmod(tmp_path, monkeypatch):
    """Reimport logging_utils with the log root redirected to a tmp dir."""
    monkeypatch.setenv("EXAONE_YAHO_ROOT", str(tmp_path))
    import logging_utils
    importlib.reload(logging_utils)
    return logging_utils, tmp_path


def test_event_writes_valid_envelope(logmod):
    m, root = logmod
    log = m.JsonlLogger(phase="train", run_id="t-1", stage="sft")
    rec = log.event("step", step=1, metrics={"loss": 1.2}, meta={"note": "x"})
    required = {"ts", "run_id", "phase", "stage", "event", "step", "metrics", "meta"}
    assert required <= rec.keys()
    assert log.path.exists()
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk["metrics"]["loss"] == 1.2
    assert on_disk["phase"] == "train"


def test_append_only(logmod):
    m, _ = logmod
    log = m.JsonlLogger(phase="eval", run_id="t-2")
    log.event("a")
    log.event("b")
    recs = list(m.iter_records(log.path))
    assert [r["event"] for r in recs] == ["a", "b"]


def test_invalid_phase_rejected(logmod):
    m, _ = logmod
    with pytest.raises(ValueError):
        m.JsonlLogger(phase="not-a-phase", run_id="t-3")


def test_manifest_and_capability_helpers(logmod):
    m, root = logmod
    m.write_manifest("t-4", seed=42, base_model="EXAONE", dtype="bfloat16")
    manifest = (root / "runs" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(manifest[-1])["run_id"] == "t-4"

    log = m.JsonlLogger(phase="eval", run_id="t-4")
    rec = log.capability(bench="KMMLU", score_before=0.50, score_after=0.47)
    assert rec["metrics"]["delta"] == pytest.approx(-0.03)


def test_non_numeric_metric_rejected(logmod):
    # §17.1 — metrics carry numeric/bool only; non-numeric belongs in meta.
    m, _ = logmod
    log = m.JsonlLogger(phase="data", run_id="t-6")
    with pytest.raises(TypeError):
        log.event("sample", metrics={"label": "not a number"})
    # numeric / bool / None are accepted
    log.event("sample", metrics={"x": 1, "y": 2.0, "flag": True, "z": None})


def test_korean_unicode_preserved(logmod):
    m, _ = logmod
    log = m.JsonlLogger(phase="data", run_id="t-5")
    log.event("sample", meta={"text": "거제 야호~"})
    raw = log.path.read_text(encoding="utf-8")
    assert "거제 야호~" in raw  # ensure_ascii=False
