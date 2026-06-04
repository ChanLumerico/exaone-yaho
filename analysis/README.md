# analysis/

Reproducible **JSONL → pandas → figure** scripts (§17.4). The JSONL logs under
`logs/` are the single source of truth; every figure must regenerate from them.

Convention: each script reads `logs/<phase>/*.jsonl` (via
`src/logging_utils.load_phase` / `iter_records`), builds a DataFrame, and writes a
figure to `reports/`. No figure should depend on transient state — re-running any
script on the committed logs reproduces the same output.

Planned figures (Phase 3): training loss/J curves, yaho/parapara firing-rate vs.
negativity/departure bins (vs. §5.3/§5.4 target curves), capability-retention
before/after (§16.1), ablation comparisons (§16.3), marker/emoji diversity.
