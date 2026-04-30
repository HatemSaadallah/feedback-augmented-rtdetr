#!/bin/bash
#SBATCH --job-name=rtdetr-abl
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/3415496/rt-detr/full_abl_%j.out
#SBATCH --error=/home/3415496/rt-detr/full_abl_%j.err

export PYTHONUNBUFFERED=1
source ~/rt-detr/notify.sh
notify "trim ablation started" "feedback-off + 480 + latency"

module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L

FB=/home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2/checkpoint.pth
BASE=/home/3415496/rt-detr/weights/rtdetr_r50vd_6x_coco.pth
CFG_FB_640=configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2.yml
CFG_BASE_640=configs/rtdetr/rtdetr_r50vd_6x_coco.yml

run () {
    local label=$1; shift
    echo ""; echo "=== ROW: $label ==="
    python -u tools/full_ablation.py "$@" --label "$label"
}

# === ESSENTIALS ONLY ===
# 1) Feedback-off ablation at 640 (PROVES mechanism contributes)
run feedback_off_640 -c "$CFG_FB_640" -r "$FB" --feedback-off

# 2) Cross-res low end (cheap, completes the 480/640/800/960 figure with the existing 640+800)
run feedback_p2_480  -c "$CFG_FB_640" -r "$FB" --resolution 480 480

# 3) Latency: ours vs baseline at 640 (validates real-time claim)
run feedback_p2_latency_640 -c "$CFG_FB_640" -r "$FB" --latency-only --latency-input-size 640
run baseline_latency_640    -c "$CFG_BASE_640" -r "$BASE" --latency-only --latency-input-size 640

echo ""; echo "=== TRIMMED ABLATION DONE ==="
ls -la /home/3415496/rt-detr/ablation_*.json | head
notify "trim ablation DONE" "results in /home/3415496/rt-detr/ablation_*.json"
