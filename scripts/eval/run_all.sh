#!/bin/zsh
cd /Users/chanlee/Desktop/Programming/exaone_yaho
PY=.venv/bin/python
echo "=== waiting for generation (pid $(cat logs/eval/generate.pid)) ==="
while kill -0 $(cat logs/eval/generate.pid) 2>/dev/null; do sleep 30; done
echo "=== [1/3] deterministic (style/firing/diversity/safety) ==="
$PY -u scripts/eval/score_det.py
echo "=== [2/3] capability (COPA + ppl) ==="
$PY -u scripts/eval/capability.py 250
echo "=== [3/3] semantic (bge-m3 Acc_style + Sim + J) ==="
$PY -u scripts/eval/semantic.py
echo "=== ALL EVAL DONE $(date '+%T') ==="
