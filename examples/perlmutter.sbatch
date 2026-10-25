#!/bin/bash
# Parity check on one NERSC Perlmutter node (4 A100s). Pass account and QOS on the command line:
#   sbatch -A <account> -q debug examples/perlmutter.sbatch <python> <outdir>
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -t 00:20:00
#SBATCH -J sprite-parity
set -euo pipefail
PY=$1
OUT=$2
PRECISION=${3:-32-true}
WORLD=${4:-4}
export HF_HOME=${HF_HOME:-$SCRATCH/huggingface}
# Submit from the repository root, or set SPRITE_EXAMPLES. Install the release checkout in the worker environment for versioned results.
HERE=${SPRITE_EXAMPLES:-$SLURM_SUBMIT_DIR/examples}
[[ -d "$HERE/../src/lightning_sprite" ]] && export PYTHONPATH="$HERE/../src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi -L
ARGS=(--model "${MODEL:-Qwen/Qwen2.5-0.5B}" --device cuda --world "$WORLD" --global-batch "$((2 * WORLD))" --precision "$PRECISION")
if [[ "$PRECISION" == bf16-mixed ]]; then
  for arm in ref32 ref split split_premul; do
    srun -n 1 --ntasks-per-node=1 "$PY" "$HERE/parity_check.py" "$arm" "$OUT" "${ARGS[@]}"
  done
else
  srun -n 1 --ntasks-per-node=1 "$PY" "$HERE/parity_check.py" ref "$OUT" "${ARGS[@]}"
fi
for arm in ddp fsdp; do
  srun -n "$WORLD" --ntasks-per-node="$WORLD" "$PY" "$HERE/parity_check.py" "$arm" "$OUT" "${ARGS[@]}"
done
status=0
"$PY" "$HERE/parity_check.py" compare "$OUT" --precision "$PRECISION" || status=$?
# Optional BF16 negative controls, e.g. FAULTS=missing-rank:wrong-scale (":" or spaces): each must be rejected.
FAULTS=${FAULTS:-}
for fault in ${FAULTS//:/ }; do
  mkdir "$OUT/fault_$fault"
  for arm in ref32 ref split split_premul; do
    ln -s "../$arm.json" "$OUT/fault_$fault/$arm.json" && ln -s "../$arm.pt" "$OUT/fault_$fault/$arm.pt"
  done
  for arm in ddp fsdp; do
    srun -n "$WORLD" --ntasks-per-node="$WORLD" "$PY" "$HERE/parity_check.py" "$arm" "$OUT/fault_$fault" "${ARGS[@]}" \
      --fault "$fault"
  done
  "$PY" "$HERE/parity_check.py" compare "$OUT/fault_$fault" --precision "$PRECISION" || true
done
# Optional DDP-only controls, with the other arms linked to the clean baseline.
DDP_FAULTS=${DDP_FAULTS:-}
for fault in ${DDP_FAULTS//:/ }; do
  case "$fault" in
    fp32-backbone | local-adapter | scale-grad | skip-update) ;;
    *) echo "Unknown DDP fault: $fault" >&2; exit 1 ;;
  esac
  mkdir "$OUT/fault_${fault}_ddp"
  for arm in ref32 ref split split_premul fsdp; do
    ln -s "../$arm.json" "$OUT/fault_${fault}_ddp/$arm.json"
    ln -s "../$arm.pt" "$OUT/fault_${fault}_ddp/$arm.pt"
  done
  srun -n "$WORLD" --ntasks-per-node="$WORLD" "$PY" "$HERE/parity_check.py" ddp "$OUT/fault_${fault}_ddp" "${ARGS[@]}" --fault "$fault"
  "$PY" "$HERE/parity_check.py" compare "$OUT/fault_${fault}_ddp" --precision "$PRECISION" || true
done
if [[ -n "$FAULTS$DDP_FAULTS" ]]; then
  "$PY" "$HERE/parity_check.py" faults "$OUT" --precision "$PRECISION" \
    --expected-faults "${FAULTS}${FAULTS:+:}${DDP_FAULTS}" || status=$?
fi
exit "$status"
