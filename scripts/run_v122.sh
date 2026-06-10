#!/bin/bash
# v12.2 FULL auto pipeline (B = data rebalance + retrain + ORPO + eval). nohup this.
# rebalance(choberi+flatten) -> mlx -> nan-guard -> SFT -> auto-select ckpt -> ORPO(+anti-rep+choberi) -> eval
cd /Users/chanlee/Desktop/Programming/exaone_yaho || exit 1
PY=.venv/bin/python
ts() { date '+%H:%M:%S'; }
echo "===== v12.2 PIPELINE START $(ts) ====="

echo "=== [1/7] rebalance corpus ($(ts)) ==="
$PY -u scripts/rebalance_markers.py || { echo "FAIL rebalance"; exit 1; }

echo "=== [2/7] build mlx + nan-guard ($(ts)) ==="
$PY scripts/corpus_to_mlx.py data/gyaru_corpus_12_2.jsonl || { echo "FAIL mlx"; exit 1; }
MAXTOK=$($PY -c "import json; from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('models/EXAONE-3.5-7.8B-Instruct-bf16',trust_remote_code=True); print(max(len(tok.apply_chat_template(json.loads(l)['messages'],tokenize=True)) for l in open('data/mlx/train.jsonl')))")
echo "max train tok = $MAXTOK (max_seq 3072)"
if [ "$MAXTOK" -ge 3072 ]; then echo "NAN-GUARD ABORT: $MAXTOK >= 3072"; exit 1; fi

echo "=== [3/7] SFT v12.2 (~3.5h) ($(ts)) ==="
$PY -u -m mlx_lm lora --model models/EXAONE-3.5-7.8B-Instruct-bf16 --train \
  --data data/mlx -c configs/sft_v122.yaml 2>&1 | tee logs/train/sft_v122.log
test -f adapters/sft_v122/adapters.safetensors || { echo "FAIL SFT"; exit 1; }

echo "=== [4/7] auto-select checkpoint ($(ts)) ==="
$PY scripts/select_ckpt.py logs/train/sft_v122.log adapters/sft_v122 1830 || { echo "FAIL select"; exit 1; }

echo "=== [5/7] build v12.2 ORPO pairs ($(ts)) ==="
$PY -u scripts/build_orpo_v122.py || { echo "FAIL orpo-pairs"; exit 1; }

echo "=== [6/7] ORPO v12.2 (~1.8h) ($(ts)) ==="
$PY -u scripts/orpo_train.py --start adapters/sft_v122/adapters.safetensors \
  --out adapters/sft_v122_orpo --pairs data/orpo_pairs_v122.jsonl --iters 520 || { echo "FAIL orpo"; exit 1; }
cat > adapters/sft_v122_orpo/SELECTED.txt <<EOF
v12.2-ORPO: start=sft_v122 | pairs=orpo_pairs_v122 (209: v121 + anti-repetition + choberi-reinforce) λ0.3 lr1e-5 520it
B retry: marker-rebalanced corpus (choberi 136->258, flatten) + ORPO anti-repetition (fix 아게미자와 amplification) + choberi-reinforce. gyaru_corpus_12_2.
EOF

echo "=== [7/7] eval ($(ts)) ==="
$PY -u scripts/eval/generate.py v122orpo
$PY -u scripts/eval/score_det.py
$PY -u scripts/eval/semantic.py
$PY -u scripts/eval/diversity.py
$PY -u scripts/eval/copa_tag.py v122orpo adapters/sft_v122_orpo 250

echo "===== v12.2 PIPELINE DONE $(ts) ====="
