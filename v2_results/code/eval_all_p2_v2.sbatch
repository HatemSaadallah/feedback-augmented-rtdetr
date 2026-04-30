#!/bin/bash
#SBATCH --job-name=rtdetr-p2-v2-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/3415496/rt-detr/eval_p2_v2_%j.out
#SBATCH --error=/home/3415496/rt-detr/eval_p2_v2_%j.err

# v2 eval: the ONLY question we care about is whether feedback ON > feedback OFF
# on AP_S. full_ablation.py supports --feedback-off, so we run both with the
# same checkpoint and write comparable JSONs.

export PYTHONUNBUFFERED=1
source ~/rt-detr/notify.sh
notify "P2 v2 eval started" "job $SLURM_JOB_ID"
module load miniconda3
source activate rtdetr
cd ~/rt-detr/rtdetr_pytorch
nvidia-smi -L

CFG=configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2_v2.yml
CKPT=/home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2_v2/checkpoint.pth

echo "=== v2 feedback ON @ 640 ==="
python -u tools/full_ablation.py -c $CFG -r $CKPT \
    --label v2_feedback_on_640 \
    --out-json /home/3415496/rt-detr/ablation_v2_feedback_on_640.json 2>&1 | tee /home/3415496/rt-detr/eval_p2_v2_on_640.log | grep -E "Average Precision|Average Recall|inference_ms|n_params|AP " | head -20

echo ""
echo "=== v2 feedback OFF @ 640 ==="
python -u tools/full_ablation.py -c $CFG -r $CKPT --feedback-off \
    --label v2_feedback_off_640 \
    --out-json /home/3415496/rt-detr/ablation_v2_feedback_off_640.json 2>&1 | tee /home/3415496/rt-detr/eval_p2_v2_off_640.log | grep -E "Average Precision|Average Recall|AP " | head -20

echo ""
echo "=== v2 feedback ON @ 800 ==="
python -u tools/full_ablation.py -c $CFG -r $CKPT \
    --resolution 800 800 \
    --label v2_feedback_on_800 \
    --out-json /home/3415496/rt-detr/ablation_v2_feedback_on_800.json 2>&1 | tee /home/3415496/rt-detr/eval_p2_v2_on_800.log | grep -E "Average Precision|Average Recall|AP " | head -20

echo ""
echo "=== V2 SUCCESS-METRIC SUMMARY ==="
python3 -c "
import json
on  = json.load(open('/home/3415496/rt-detr/ablation_v2_feedback_on_640.json'))
off = json.load(open('/home/3415496/rt-detr/ablation_v2_feedback_off_640.json'))
on8 = json.load(open('/home/3415496/rt-detr/ablation_v2_feedback_on_800.json'))
def f(x): return f'{(x or 0)*100:.2f}' if x is not None else '   - '
print(f'                        AP        AP_S       AP_M       AP_L')
print(f'  v2 ON  @ 640:    {f(on[\"AP\"])}     {f(on[\"AP_S\"])}      {f(on[\"AP_M\"])}      {f(on[\"AP_L\"])}')
print(f'  v2 OFF @ 640:    {f(off[\"AP\"])}     {f(off[\"AP_S\"])}      {f(off[\"AP_M\"])}      {f(off[\"AP_L\"])}')
print(f'  v2 ON  @ 800:    {f(on8[\"AP\"])}     {f(on8[\"AP_S\"])}      {f(on8[\"AP_M\"])}      {f(on8[\"AP_L\"])}')
print()
delta_aps = (on['AP_S'] - off['AP_S']) * 100
print(f'  delta AP_S (ON - OFF) = {delta_aps:+.2f}   {\"PASS  feedback works\" if delta_aps > 0.3 else \"FAIL  no causal contribution\"}')
"

notify "P2 v2 eval DONE" "results in eval_p2_v2_$SLURM_JOB_ID.out"
