#!/bin/bash
#SBATCH --job-name=rtdetr-viz
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3415496/rt-detr/viz_%j.out
#SBATCH --error=/home/3415496/rt-detr/viz_%j.err

source ~/rt-detr/notify.sh
notify "viz started" "job $SLURM_JOB_ID"
module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L

mkdir -p ~/rt-detr/viz_out
CKPT=/home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_finetune/checkpoint.pth
CFG=configs/rtdetr/rtdetr_r50vd_hpc_feedback_finetune.yml

for img in 000000000139.jpg 000000000632.jpg 000000000785.jpg 000000001000.jpg 000000002006.jpg; do
  echo "=== viz on $img ==="
  python tools/visualize_feedback.py \
      --baseline-config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
      --baseline-ckpt   /home/3415496/rt-detr/weights/rtdetr_r50vd_6x_coco.pth \
      --feedback-config $CFG \
      --feedback-ckpt   $CKPT \
      --image           /home/3415496/rt-detr/data/coco/val2017/$img \
      --out             /home/3415496/rt-detr/viz_out/viz_${img%.jpg}.png \
      --device cuda --score-thr 0.4 || echo "viz failed for $img"
done
ls -la ~/rt-detr/viz_out/
notify "viz done" "PNGs at ~/rt-detr/viz_out/"
