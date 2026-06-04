"""§17 — Unified append-only JSONL metric logging (the project's SSOT).

Every loggable signal across the pipeline (data synthesis / training / eval /
inference / embeddings / system) is written here as one-line-per-event JSONL.
JSONL is the single source of truth for *every* figure and analysis: any plot
must be regenerable from these files (§17.4). "An unrecorded metric is a metric
that does not exist."

Common envelope (§17.1) — one line == one event::

    {"ts": ISO-8601, "run_id": "<uuid|name>",
     "phase": "data|train|eval|infer", "stage": "synth|sft|dpo|...",
     "event": "<event_type>", "step": <int|null>,
     "metrics": {...},   # numeric / boolean signals
     "meta":    {...}}   # non-numeric identifiers / labels / context

Conventions (§17.4):
  * Location: ``logs/<phase>/<run_id>.jsonl``  (a new run == a new file)
  * Append-only, never overwrite. UTF-8. ``ts`` is ISO-8601.
  * All writes go through this single utility so the schema is enforced.
  * Large vectors never go inline -> store in ``logs/embeddings/<run_id>.parquet``
    (or ``.npy``) and log only an index record (§17.2).

This module is intentionally dependency-light (stdlib only) so the logging
skeleton works from Phase 0. Parquet helpers import numpy/pandas lazily.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Paths / roots
# --------------------------------------------------------------------------- #

# Repo root = parent of this file's directory (src/..). Overridable via env so
# tests / notebooks can redirect writes.
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def _root() -> Path:
    return Path(os.environ.get("EXAONE_YAHO_ROOT", str(_DEFAULT_ROOT)))


def logs_dir() -> Path:
    return _root() / "logs"


def runs_dir() -> Path:
    return _root() / "runs"


def embeddings_dir() -> Path:
    return logs_dir() / "embeddings"


VALID_PHASES = {"data", "train", "eval", "infer", "system"}


def new_run_id(prefix: str | None = None) -> str:
    """Short unique run id, optionally namespaced (e.g. ``sft-3f9a1c2b``)."""
    short = uuid.uuid4().hex[:8]
    return f"{prefix}-{short}" if prefix else short


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_append(path: Path, line: str) -> None:
    """Append a single line. ``a`` mode + per-line write is atomic enough for
    our append-only, single-writer-per-file model; a lock guards intra-process
    concurrency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


_WRITE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #


@dataclass
class JsonlLogger:
    """Append-only JSONL logger bound to one (phase, run_id).

    Example::

        log = JsonlLogger(phase="train", run_id=new_run_id("sft"), stage="sft")
        log.event("step", step=10, metrics={"loss": 1.23, "lr": 1e-4})
        log.event("checkpoint", step=200,
                  metrics={"val_loss": 1.1, "J": 0.42},
                  meta={"ckpt": "adapters/0200"})
    """

    phase: str
    run_id: str
    stage: str | None = None
    _path: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.phase not in VALID_PHASES:
            raise ValueError(
                f"phase must be one of {sorted(VALID_PHASES)}, got {self.phase!r}"
            )
        self._path = logs_dir() / self.phase / f"{self.run_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def event(
        self,
        event: str,
        *,
        metrics: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        step: int | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        """Write one enveloped event line and return the record dict."""
        record = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "phase": self.phase,
            "stage": stage if stage is not None else self.stage,
            "event": event,
            "step": step,
            "metrics": metrics or {},
            "meta": meta or {},
        }
        _validate_envelope(record)
        _atomic_append(self._path, json.dumps(record, ensure_ascii=False))
        return record

    # Convenience wrappers for the most common record kinds (§17.2) ---------- #

    def step(self, step: int, **metrics: Any) -> dict[str, Any]:
        """Training step record: loss/lr/grad_norm/tok_per_s/... (§17.2 train)."""
        return self.event("step", step=step, metrics=metrics)

    def sample(
        self, *, metrics: dict[str, Any], meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Per-sample record (data synth §17.2 / eval example §17.2)."""
        return self.event("sample", metrics=metrics, meta=meta)

    def curve(self, *, bin: Any, mean_score: float, firing_rate: float, kind: str) -> dict[str, Any]:
        """Gating-curve point for §5.3/§5.4 firing-rate validation (§17.2)."""
        return self.event(
            "curve",
            metrics={"mean_score": mean_score, "firing_rate": firing_rate},
            meta={"bin": bin, "kind": kind},  # kind: "yaho" | "parapara"
        )

    def capability(
        self, *, bench: str, score_before: float, score_after: float
    ) -> dict[str, Any]:
        """Capability-retention record (§16.1)."""
        return self.event(
            "capability",
            metrics={
                "score_before": score_before,
                "score_after": score_after,
                "delta": score_after - score_before,
            },
            meta={"bench": bench},
        )

    def aggregate(self, **metrics: Any) -> dict[str, Any]:
        """Corpus/eval aggregate record."""
        return self.event("aggregate", metrics=metrics)


# --------------------------------------------------------------------------- #
# Run manifest (§17.3) — reproducibility index
# --------------------------------------------------------------------------- #


def write_manifest(
    run_id: str,
    *,
    git_commit: str | None = None,
    config_snapshot: str | dict | None = None,
    seed: int | None = None,
    base_model: str | None = None,
    dataset_version: str | None = None,
    dtype: str | None = None,
    hardware: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a one-time manifest row to ``runs/manifest.jsonl`` so every figure
    is traceable to code + config + seed (§17.3)."""
    record = {
        "ts": _utc_now_iso(),
        "run_id": run_id,
        "git_commit": git_commit,
        "config_snapshot": config_snapshot,
        "seed": seed,
        "base_model": base_model,
        "dataset_version": dataset_version,
        "dtype": dtype,
        "hardware": hardware,
    }
    if extra:
        record.update(extra)
    _atomic_append(runs_dir() / "manifest.jsonl", json.dumps(record, ensure_ascii=False))
    return record


# --------------------------------------------------------------------------- #
# Embedding store (§17.2) — vectors out-of-band, index inline
# --------------------------------------------------------------------------- #


def log_embeddings(
    logger: JsonlLogger,
    *,
    vectors: "Iterable[Iterable[float]]",
    labels: list[Any],
    model: str,
    purpose: str,
) -> Path:
    """Persist vectors to ``logs/embeddings/<run_id>.parquet`` and write one
    index record per row to JSONL (``emb_id, row, model, purpose, label``).

    Returns the parquet path. Lazily imports numpy/pandas; raises a clear error
    if unavailable.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "log_embeddings requires numpy + pandas (pyarrow for parquet). "
            "Install the analysis extras."
        ) from exc

    arr = np.asarray(list(vectors), dtype="float32")
    if arr.shape[0] != len(labels):
        raise ValueError("vectors and labels length mismatch")

    embeddings_dir().mkdir(parents=True, exist_ok=True)
    out = embeddings_dir() / f"{logger.run_id}.parquet"
    df = pd.DataFrame(arr)
    df.columns = [f"d{i}" for i in range(arr.shape[1])]
    df.insert(0, "label", labels)
    df.to_parquet(out, index=True)

    for row, label in enumerate(labels):
        logger.event(
            "embedding",
            meta={
                "emb_id": f"{logger.run_id}:{row}",
                "row": row,
                "model": model,
                "purpose": purpose,
                "label": label,
                "parquet": str(out.relative_to(_root())),
            },
        )
    return out


# --------------------------------------------------------------------------- #
# Reading back (analysis helpers, §17.4)
# --------------------------------------------------------------------------- #


def iter_records(path: str | Path) -> "Iterable[dict[str, Any]]":
    """Yield records from a JSONL log file (skips blank lines)."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_phase(phase: str, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load all records for a phase across every run file (for pandas/plots)."""
    base = (Path(root) if root else logs_dir()) / phase
    out: list[dict[str, Any]] = []
    if not base.exists():
        return out
    for f in sorted(base.glob("*.jsonl")):
        out.extend(iter_records(f))
    return out


# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #

_REQUIRED_KEYS = {"ts", "run_id", "phase", "stage", "event", "step", "metrics", "meta"}


def _validate_envelope(record: dict[str, Any]) -> None:
    missing = _REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"log record missing required keys: {sorted(missing)}")
    if not isinstance(record["metrics"], dict) or not isinstance(record["meta"], dict):
        raise TypeError("'metrics' and 'meta' must be dicts")
    # §17.1 — metrics carry numeric/boolean signals only; non-numeric context
    # (labels, ids, text) belongs in meta. Enforced so figures can load metrics
    # straight into numeric columns. (bool is a subclass of int.)
    for key, val in record["metrics"].items():
        if val is not None and not isinstance(val, (int, float)):
            raise TypeError(
                f"metric {key!r} must be numeric/bool/None (got {type(val).__name__}); "
                "put non-numeric data in 'meta'"
            )
