#!/bin/bash
#SBATCH --job-name=rtdetr-rerun
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3415496/rt-detr/eval_rerun_%j.out
#SBATCH --error=/home/3415496/rt-detr/eval_rerun_%j.err

export PYTHONUNBUFFERED=1
source ~/rt-detr/notify.sh
notify "rerun started" "feedback_p2_800 + sliced_eval"
module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L

FB=/home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2/checkpoint.pth

echo ""; echo "=== feedback_p2_800 ==="
python -u tools/train.py -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_eval800.yml -r "$FB" --test-only 2>&1 | tee /home/3415496/rt-detr/eval_p2_feedback_p2_800.log | grep -E "Average Precision|Average Recall"

echo ""; echo "=== feedback_p2_sliced (640 patches, 0.2 overlap) ==="
python -u tools/sliced_eval.py -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2.yml -r "$FB" --slice-size 640 --overlap 0.2 2>&1 | tee /home/3415496/rt-detr/eval_p2_feedback_sliced.log | grep -E "Average Precision|Average Recall|preds="

notify "rerun DONE" "results in eval_rerun_$SLURM_JOB_ID.out"
