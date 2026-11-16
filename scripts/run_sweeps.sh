#!/usr/bin/env bash
# Threshold sweep of every trained model on the OTHER city (inference only, ~1 min per model on a T4).
#
#   DATA=/content/data OUT=/content/drive/MyDrive/ccbs_runs bash scripts/run_sweeps.sh
#
# Writes results/sweeps/<run>_on_<city>.json; then run scripts/make_figures.py.
set -euo pipefail
DATA="${DATA:-data}"
OUT="${OUT:-runs}"
mkdir -p results/sweeps
shopt -s nullglob
CKPTS=("$OUT"/*_global_seed*/)
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "ERROR: no checkpoints found in $OUT (is Google Drive mounted? does the folder exist?)" >&2
  exit 1
fi
for CKPT_DIR in "${CKPTS[@]}"; do
  RUN=$(basename "$CKPT_DIR")
  SRC=${RUN%%_*}
  TGT=$([ "$SRC" = paris ] && echo khartoum || echo paris)
  python scripts/evaluate.py --data-root "$DATA" --ckpt "$CKPT_DIR/best.pt" --city "$TGT" \
    --split all --sweep --out "results/sweeps/${RUN}_on_${TGT}.json" > /dev/null
  echo "done ${RUN} on ${TGT}"
done
