#!/bin/bash
#SBATCH --job-name=rtdetr-p2-v2-chain
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3415496/rt-detr/p2_v2_chain_%j.out
#SBATCH --error=/home/3415496/rt-detr/p2_v2_chain_%j.err

# v2 chain: gate_init=0.0, gate_floor=0.1, level_mask=[T,T,F,F] (P2/P3 only).
# Output dir is separate from v1 so v1 ckpt stays untouched.

export PYTHONUNBUFFERED=1
source ~/rt-detr/notify.sh
notify "P2 v2 chain link started" "job $SLURM_JOB_ID. Checking checkpoint..."

module load miniconda3
source activate rtdetr

CONFIG=/home/3415496/rt-detr/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2_v2.yml
OUT_DIR=/home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2_v2
CKPT=$OUT_DIR/checkpoint.pth
TARGET_EPOCHS=12
SBATCH_SELF=~/rt-detr/rtdetr_pytorch/scripts_hpc/train_p2_v2_chain.sbatch
EVAL_SBATCH=~/rt-detr/rtdetr_pytorch/scripts_hpc/eval_all_p2_v2.sbatch

mkdir -p $OUT_DIR

# Stage train2017 to compute-node /tmp; symlink home/.../train2017 to it.
STAGE=${TMPDIR:-/tmp}/coco_$SLURM_JOB_ID
mkdir -p $STAGE
echo "=== unzipping train2017 from scratch zip to $STAGE ==="
unzip -q /mnt/beegfsstudents/scratch_3415496/train2017.zip -d $STAGE/
echo "staged: $(ls $STAGE/train2017 | wc -l) train images"

ln -sfn $STAGE/train2017 ~/rt-detr/data/coco/train2017
echo "symlinked ~/rt-detr/data/coco/train2017 -> $STAGE/train2017"
ls -la ~/rt-detr/data/coco/train2017 | head -1

cleanup () {
    rm -f ~/rt-detr/data/coco/train2017
    rm -rf "$STAGE"
}
trap cleanup EXIT

cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L

if [ -f "$CKPT" ]; then
    echo "=== tuning from $CKPT (continuing v2 chain) ==="
    TRAIN_FLAG="--tuning $CKPT"
else
    echo "=== fresh tuning from RT-DETR baseline ==="
    TRAIN_FLAG="--tuning /home/3415496/rt-detr/weights/rtdetr_r50vd_6x_coco.pth"
fi

python -u tools/train.py \
    -c $CONFIG \
    --amp \
    $TRAIN_FLAG
rc=$?
echo "training exit=$rc"

LOG_FILE=$OUT_DIR/log.txt
TOTAL_EPOCHS=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-0}
echo "total epochs done: $TOTAL_EPOCHS (target $TARGET_EPOCHS)"

if [ "$TOTAL_EPOCHS" -ge "$TARGET_EPOCHS" ]; then
    notify "P2 v2 training DONE" "Reached $TARGET_EPOCHS epochs. Submitting eval_all_p2_v2."
    sbatch "$EVAL_SBATCH"
elif [ "$rc" = "0" ] || [ "$rc" = "143" ] || [ "$rc" = "137" ]; then
    notify "P2 v2 chain link end" "epoch $TOTAL_EPOCHS/$TARGET_EPOCHS, resubmitting"
    sbatch "$SBATCH_SELF"
else
    notify "P2 v2 chain FAILED" "exit=$rc at epoch $TOTAL_EPOCHS"
fi
exit 0
