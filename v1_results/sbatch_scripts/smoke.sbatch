#!/bin/bash
#SBATCH --job-name=rtdetr-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/3415496/rt-detr/smoke_%j.out
#SBATCH --error=/home/3415496/rt-detr/smoke_%j.err

module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L
python tools/smoke_test_feedback.py
