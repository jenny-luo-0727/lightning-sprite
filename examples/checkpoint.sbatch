#!/bin/bash
# Checkpoint save/resume check on one NERSC Perlmutter node (4 A100s). Pass account and QOS on the command line:
#   sbatch -A <account> -q debug examples/checkpoint.sbatch <python> <outdir> [precision]
# STAGE=train runs every role except the fault controls, STAGE=faults runs the fault controls and the comparison
# (after a train stage into the same outdir), and STAGE=all (the default) runs both.
# Adjust the walltime for your model, hardware and selected stage.
# The outdir holds two checkpoints per strategy arm; storage depends on the selected model.
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -J sprite-checkpoint
set -euo pipefail
PY=$1
OUT=$2
PRECISION=${3:-32-true}
STAGE=${STAGE:-all}
case "$STAGE" in
  all) PARTS="train faults" ;;
  train | faults) PARTS=$STAGE ;;
  *) echo "STAGE must be all, train or faults, not '$STAGE'" >&2; exit 1 ;;
esac
# One claim per part, so a second submission into the same outdir stops here instead of racing the first.
mkdir -p "$OUT"
for part in $PARTS; do mkdir "$OUT/.claim-$part"; done
WORLD=4
export HF_HOME=${HF_HOME:-$SCRATCH/huggingface}
# Submit from the repository root, or set SPRITE_EXAMPLES. Install the release checkout in the worker environment for versioned results.
HERE=${SPRITE_EXAMPLES:-$SLURM_SUBMIT_DIR/examples}
[[ -d "$HERE/../src/lightning_sprite" ]] && export PYTHONPATH="$HERE/../src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi -L
ARGS=(--model "${MODEL:-Qwen/Qwen2.5-0.5B}" --device cuda --global-batch "$((2 * WORLD))" --precision "$PRECISION")
# launch <processes> <arm> <role> [extra args]
launch() {
  local n=$1 arm=$2 role=$3
  shift 3
  srun -n "$n" --ntasks-per-node="$n" "$PY" "$HERE/checkpoint_check.py" run "$arm" "$role" "$OUT" "${ARGS[@]}" \
    --world "$n" "$@"
}
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
  for role in full twin save resume reload; do
    launch 1 single "$role"
    for arm in ddp fsdp fsdp_file; do launch "$WORLD" "$arm" "$role"; done
  done
  launch "$((WORLD / 2))" fsdp reshard
  for source in ddp fsdp_file; do launch 1 single export --source "$source"; done
fi
if [[ "$STAGE" == all || "$STAGE" == faults ]]; then
  for fault in drop-optimizer drop-scheduler perturb-weight; do
    launch 1 single resume --fault "$fault"
    for arm in ddp fsdp fsdp_file; do launch "$WORLD" "$arm" resume --fault "$fault"; done
  done
  "$PY" "$HERE/checkpoint_check.py" compare "$OUT"
fi
