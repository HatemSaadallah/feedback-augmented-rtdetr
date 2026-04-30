#!/bin/bash
#SBATCH --job-name=rtdetr-p2-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/rt-detr/eval_p2_%j.out
#SBATCH --error=/home/3415496/rt-detr/eval_p2_%j.err

export PYTHONUNBUFFERED=1
source ~/rt-detr/notify.sh
notify "P2 eval started" "job $SLURM_JOB_ID"
module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L

BASE=/home/3415496/rt-detr/weights/rtdetr_r50vd_6x_coco.pth
FB=/home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2/checkpoint.pth

run_eval () {
  local name="$1"; local cfg="$2"; local ckpt="$3"
  echo ""
  echo "=========================================="
  echo " $name | cfg=$cfg ckpt=$ckpt"
  echo "=========================================="
  python -u tools/train.py -c "$cfg" -r "$ckpt" --test-only 2>&1 | \
      tee "/home/3415496/rt-detr/eval_p2_${name}.log" | \
      grep -E "Average Precision|Average Recall" | head -12
}

run_eval baseline_640 configs/rtdetr/rtdetr_r50vd_6x_coco.yml                "$BASE"
run_eval baseline_800 configs/rtdetr/rtdetr_r50vd_hpc_baseline_eval800.yml   "$BASE"
run_eval feedback_p2_640 configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2.yml     "$FB"
run_eval feedback_p2_800 configs/rtdetr/rtdetr_r50vd_hpc_feedback_eval800.yml "$FB"

# Sliced (SAHI-style) inference at slice_size=640, 20% overlap. Targets small objects.
echo ""; echo "=== sliced_eval (SAHI-style) ==="
python -u tools/sliced_eval.py -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2.yml -r "$FB" --slice-size 640 --overlap 0.2 2>&1 | tee /home/3415496/rt-detr/eval_p2_feedback_sliced.log | grep -E "Average Precision|Average Recall"

# Build a final summary
echo ""
echo "=== P2 ABLATION SUMMARY ==="
for v in baseline_640 baseline_800 feedback_p2_640; do
  AP=$(grep "area=   all | maxDets=100" /home/3415496/rt-detr/eval_p2_${v}.log | head -1 | awk "{print \$NF}")
  APS=$(grep "area= small" /home/3415496/rt-detr/eval_p2_${v}.log | head -1 | awk "{print \$NF}")
  APM=$(grep "area=medium" /home/3415496/rt-detr/eval_p2_${v}.log | head -1 | awk "{print \$NF}")
  APL=$(grep "area= large" /home/3415496/rt-detr/eval_p2_${v}.log | head -1 | awk "{print \$NF}")
  echo "$v: AP=$AP AP_S=$APS AP_M=$APM AP_L=$APL"
done

notify "P2 eval DONE" "results in eval_p2_$SLURM_JOB_ID.out"
