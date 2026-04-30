#!/bin/bash
#SBATCH --job-name=rtdetr-fb-resume
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/3415496/rt-detr/train_resume_%j.out
#SBATCH --error=/home/3415496/rt-detr/train_resume_%j.err

export PYTHONUNBUFFERED=1
source ~/rt-detr/notify.sh
notify "RT-DETR resume started" "job $SLURM_JOB_ID. Resuming from epoch 3 with multi-scale."

module load miniconda3
source activate rtdetr

STAGE=${TMPDIR:-/tmp}/coco_$SLURM_JOB_ID
mkdir -p $STAGE
ln -sfn ~/rt-detr/data/coco/val2017 $STAGE/val2017
ln -sfn ~/rt-detr/data/coco/annotations $STAGE/annotations
unzip -q ~/rt-detr/data/coco/train2017.zip -d $STAGE/ 2>/dev/null || ln -sfn ~/rt-detr/data/coco/train2017 $STAGE/train2017
echo "staged: $(ls $STAGE/train2017 | wc -l) train / $(ls $STAGE/val2017 | wc -l) val"

CFG=~/rt-detr/rtdetr_pytorch/configs/dataset/coco_detection.yml
ORIG=$(mktemp)
cp "$CFG" "$ORIG"
sed -i "s|/home/3415496/rt-detr/data/coco|$STAGE|g" "$CFG"
cleanup () { cp "$ORIG" "$CFG"; rm -f "$ORIG"; rm -rf "$STAGE"; }
trap cleanup EXIT

cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L
python -u tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_resume.yml \
    --amp \
    --resume /home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_finetune/checkpoint.pth
rc=$?
echo "training exit=$rc"

if [ "$rc" = "0" ]; then
  notify "RT-DETR resume finished" "job $SLURM_JOB_ID done. Submitting eval_all..."
  sbatch ~/rt-detr/rtdetr_pytorch/scripts_hpc/eval_all.sbatch
else
  notify "RT-DETR resume FAILED" "job $SLURM_JOB_ID exit=$rc"
fi
exit $rc
