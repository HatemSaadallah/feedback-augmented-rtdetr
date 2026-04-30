#!/bin/bash
#SBATCH --job-name=rtdetr-abl-viz
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3415496/rt-detr/abl_viz_%j.out
#SBATCH --error=/home/3415496/rt-detr/abl_viz_%j.err

source ~/rt-detr/notify.sh
notify "ablation+viz started" "job $SLURM_JOB_ID"

module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L

CKPT_DIR=/home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_finetune
CFG640=configs/rtdetr/rtdetr_r50vd_hpc_feedback_finetune.yml
CFG800=configs/rtdetr/rtdetr_r50vd_hpc_feedback_eval800.yml

run_eval () {
  local ckpt="$1"; local cfg="$2"; local label="$3"
  echo ""
  echo "=========================================="
  echo " EVAL $label  ckpt=$ckpt  cfg=$cfg"
  echo "=========================================="
  python tools/train.py -c "$cfg" -r "$ckpt" --test-only 2>&1 | \
      tee "/home/3415496/rt-detr/abl_${label}.log" | \
      grep -E "Average Precision|Average Recall|number of params" | head -15
}

# === ablation: each saved checkpoint at both resolutions ===
for e in 0000 0001 0002 0003; do
  run_eval "$CKPT_DIR/checkpoint${e}.pth" "$CFG640" "ep${e}_640"
  run_eval "$CKPT_DIR/checkpoint${e}.pth" "$CFG800" "ep${e}_800"
done

# === visualization: side-by-side baseline vs feedback + S3 heatmap on 3 sample images ===
mkdir -p ~/rt-detr/viz_out
for img in 000000000139.jpg 000000000632.jpg 000000000785.jpg; do
  echo "=== viz on $img ==="
  python tools/visualize_feedback.py \
      --baseline-config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
      --baseline-ckpt   /home/3415496/rt-detr/weights/rtdetr_r50vd_6x_coco.pth \
      --feedback-config "$CFG640" \
      --feedback-ckpt   "$CKPT_DIR/checkpoint.pth" \
      --image           /home/3415496/rt-detr/data/coco/val2017/$img \
      --out             /home/3415496/rt-detr/viz_out/viz_${img%.jpg}.png \
      --device cuda --score-thr 0.4 || echo "viz failed for $img"
done

# === ablation summary table ===
echo ""
echo "=== ABLATION SUMMARY (AP/AP_S/AP_M/AP_L) ==="
printf "%-20s %8s %8s %8s %8s\n" ckpt+res AP AP_S AP_M AP_L
for f in /home/3415496/rt-detr/abl_*.log; do
  label=$(basename "$f" .log | sed "s/abl_//")
  AP=$(grep "area=   all | maxDets=100" "$f" | head -1 | awk "{print \$NF}")
  APS=$(grep "area= small" "$f" | head -1 | awk "{print \$NF}")
  APM=$(grep "area=medium" "$f" | head -1 | awk "{print \$NF}")
  APL=$(grep "area= large" "$f" | head -1 | awk "{print \$NF}")
  printf "%-20s %8s %8s %8s %8s\n" "$label" "$AP" "$APS" "$APM" "$APL"
done

notify "ablation+viz done" "job $SLURM_JOB_ID complete. Check ~/rt-detr/abl_viz_$SLURM_JOB_ID.out and ~/rt-detr/viz_out/"
