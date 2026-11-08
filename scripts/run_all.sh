#!/usr/bin/env bash
# Run the full experiment grid. Every step is resumable / skipped when already done, so it is
# safe to re-run after a Colab disconnect.
#
#   DATA=/content/data OUT=/content/drive/MyDrive/ccbs_runs bash scripts/run_all.sh
#
# Steps: (1) train 2 source cities x SEEDS, (2) threshold sweeps on the other city,
#        (3) few-shot adaptation (Paris -> Khartoum by default), (4) figures + summary.
# Rough cost on a Colab T4 (batch 4, 8 epochs): 20-35 min per training run (6 runs ~ 2-3.5 h),
# ~1 min per sweep, ~45 min for the few-shot sweep (the N=200 point dominates).
#
# Environment variables: DATA, OUT, SEEDS ("0 1 2"), EPOCHS (8), TARGETS ("khartoum"; add "paris"
# for both directions), N_IMAGES ("0 10 50 200").
set -euo pipefail

DATA="${DATA:-data}"
OUT="${OUT:-runs}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-8}"
TARGETS="${TARGETS:-khartoum}"
N_IMAGES="${N_IMAGES:-0 10 50 200}"

mkdir -p results/runs

# (1) training
for SEED in $SEEDS; do
  for CITY in paris khartoum; do
    NAME="${CITY}_global_seed${SEED}"
    if [ -f "results/runs/${NAME}.json" ]; then echo "skip ${NAME} (done)"; continue; fi
    python scripts/train.py --data-root "$DATA" --out-dir "$OUT" \
      --train-city "$CITY" --seed "$SEED" --epochs "$EPOCHS"
  done
done

# (2) threshold sweeps (inference only)
for CKPT_DIR in "$OUT"/*_global_seed*/; do
  RUN=$(basename "$CKPT_DIR")
  SRC=${RUN%%_*}
  TGT=$([ "$SRC" = paris ] && echo khartoum || echo paris)
  F="results/sweeps/${RUN}_on_${TGT}.json"
  if [ -f "$F" ]; then echo "skip sweep ${RUN}"; continue; fi
  mkdir -p results/sweeps
  python scripts/evaluate.py --data-root "$DATA" --ckpt "$CKPT_DIR/best.pt" --city "$TGT" \
    --split all --sweep --out "$F" > /dev/null
  echo "sweep done: ${RUN} on ${TGT}"
done

# (3) few-shot: the source model is the seed-0 model of the other city; control = ImageNet only
for TARGET in $TARGETS; do
  SOURCE=$([ "$TARGET" = khartoum ] && echo paris || echo khartoum)
  for INIT in "$OUT/${SOURCE}_global_seed0/best.pt" imagenet; do
    NAME=$([ "$INIT" = imagenet ] && echo imagenet || echo source)
    F="results/runs/fewshot_${TARGET}_${NAME}.json"
    if [ -f "$F" ]; then echo "skip few-shot ${TARGET}/${NAME} (done)"; continue; fi
    python scripts/few_shot.py --data-root "$DATA" --target-city "$TARGET" \
      --init-ckpt "$INIT" --n-images $N_IMAGES --seeds 0 1 2 --out "$F"
  done
done

# (4) figures + summary
python scripts/make_figures.py
