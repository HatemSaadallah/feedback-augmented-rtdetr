#!/bin/bash
#SBATCH --job-name=rtdetr-eval-all
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/rt-detr/eval_all_%j.out
#SBATCH --error=/home/3415496/rt-detr/eval_all_%j.err

source ~/rt-detr/notify.sh
notify "RT-DETR eval-all started" "job $SLURM_JOB_ID. Evaluating baseline vs feedback at 640 and 800."

module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch

BASE=~/rt-detr/weights/rtdetr_r50vd_6x_coco.pth
FB=~/rt-detr/output/rtdetr_r50vd_hpc_feedback_finetune/checkpoint.pth

run_one () {
    local name="$1"; local cfg="$2"; local ckpt="$3"
    echo ""
    echo "=========================================="
    echo " $name | cfg=$cfg ckpt=$ckpt"
    echo "=========================================="
    python tools/train.py -c "$cfg" -r "$ckpt" --test-only 2>&1 | \
        tee "/home/3415496/rt-detr/eval_${name}.log" | \
        grep -E "Average Precision|Average Recall|number of params"
}

nvidia-smi -L
run_one baseline_640 configs/rtdetr/rtdetr_r50vd_6x_coco.yml                     "$BASE"
run_one baseline_800 configs/rtdetr/rtdetr_r50vd_hpc_baseline_eval800.yml        "$BASE"
run_one feedback_640 configs/rtdetr/rtdetr_r50vd_hpc_feedback_finetune.yml        "$FB"
run_one feedback_800 configs/rtdetr/rtdetr_r50vd_hpc_feedback_eval800.yml         "$FB"

echo ""
echo "=== SUMMARY ==="
SUMMARY=""
for v in baseline_640 baseline_800 feedback_640 feedback_800; do
  echo ""
  echo "--- $v ---"
  grep -E "Average Precision|Average Recall" "/home/3415496/rt-detr/eval_${v}.log" | head -12
  APS=$(grep "area= small" "/home/3415496/rt-detr/eval_${v}.log" | head -1 | awk "{print \$NF}")
  AP=$(grep "IoU=0.50:0.95 | area=   all | maxDets=100" "/home/3415496/rt-detr/eval_${v}.log" | head -1 | awk "{print \$NF}")
  SUMMARY="$SUMMARY $v: AP=$AP AP_S=$APS ;"
done
notify "RT-DETR eval-all finished" "Results: $SUMMARY"
