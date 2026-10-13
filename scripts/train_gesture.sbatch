#!/bin/bash
#SBATCH --job-name=transgesture
#SBATCH --partition=HCESC-L40S-normal,eng-research-gpu
#SBATCH --account=jrehg-cs-eng
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/xucao2/TransGesture/logs/%x_%j.out

set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=${CONFIG:-configs/gesture_jointcrossfusion_vit_base.py}
NAME=$(basename "$CONFIG" .py)
REPO=/projects/illinois/eng/cs/jrehg/users/xucao2/TransGesture
PY=/projects/illinois/eng/cs/jrehg/users/xucao2/miniconda3/envs/anygaze/bin/python
cd "$REPO"

echo "host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG smoke=${SMOKE:-0}"
nvidia-smi -L
$PY -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"

OPTS=()
OUT_DIR=./output/$NAME
if [ "${SMOKE:-0}" = "1" ]; then
    OUT_DIR=./output/smoke_$NAME
    OPTS+=(train.max_iter=20 train.log_period=1 train.checkpointer.period=20)
fi
OPTS+=(train.output_dir=$OUT_DIR)

$PY -u tools/train.py --config-file "$CONFIG" --num-gpus 1 "${OPTS[@]}"

$PY -u tools/eval_on_gesture_target.py --config_file "$CONFIG" --model_weights "$OUT_DIR/model_final.pth" \
    | tee "$OUT_DIR/eval_test.txt"
