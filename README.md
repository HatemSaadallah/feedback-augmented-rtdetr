# Feedback-Augmented RT-DETR

> A cross-attention refinement strategy for enhanced small-object detection.
> Computer Vision and Image Processing, Bocconi University.

This repository extends [RT-DETR](https://github.com/lyuwenyu/RT-DETR) with a
**decoder-to-encoder feedback module** that refines encoder memory mid-decoding
using preliminary predictions from the first decoder layer. Later decoder layers
then attend over the refined memory.

A first implementation (**v1**) trained cleanly but a same-checkpoint inference
ablation showed the mechanism contributed nothing on small-object AP
(ΔAP\_S = 0.00). We diagnosed gate collapse during training and re-implemented
(**v2**) with a gate-floor reparameterization and a P2/P3-only level mask. The
v2 retrain produces a measurable causal contribution of
**ΔAP\_S = +0.99** on COCO val2017 @ 640.

The full write-up — math, multiple approaches, ablations, training details,
gate dynamics, qualitative detections — is in
[**`report/main.pdf`**](report/main.pdf) (16 pages).

---

## Headline numbers

Same-checkpoint ablation on COCO val2017 @ 640 (only `feedback.disabled` toggled):

| Mode | AP | **AP\_S** | AP\_M | AP\_L |
|---|---|---|---|---|
| v2 ON  | 52.10 | **34.25** | 56.33 | 68.82 |
| v2 OFF | 51.53 | 33.26 | 55.91 | 68.32 |
| **Δ (causal)** | **+0.57** | **+0.99** | **+0.42** | **+0.50** |

For reference, **v1**'s same ablation gave ΔAP\_S = 0.00 — the mechanism was
silent because the gate had decayed to ≈ 0.12 during training. v2 holds the
gate at ≈ 0.39 throughout training via the floor reparameterization. **Both
versions have identical parameter count (49.08 M); the v2 changes are
zero-parameter.**

At higher inference resolution (800 × 800):

| Resolution | AP | AP\_S | AP\_M | AP\_L |
|---|---|---|---|---|
| 640 | 52.10 | 34.25 | 56.33 | 68.82 |
| **800** | **52.31** | **37.01** | 56.35 | 67.04 |

---

## What changed (v1 → v2)

Three structural changes to `src/zoo/rtdetr/feedback_module.py`. **No new
parameters**; only reparameterization plus a compute-time level mask.

```python
# 1. Gate init bumped: sigmoid(-2.0)=0.12  →  sigmoid(0.0)=0.50
gate_init: -2.0  →  0.0

# 2. Gate floor reparameterization (NEW): gate cannot collapse below floor.
#    Before:  g_eff = sigmoid(alpha)
#    After:   g_eff = floor + (1 - floor) * sigmoid(alpha)
gate_floor: 0.1

# 3. Level mask (NEW): refine only P2/P3 features (where small objects live).
#    Cross-attention computed on memory subset; S4/S5 token positions
#    in the refined memory are byte-identical to the input.
level_mask: [True, True, False, False]  # (P2, P3, P4, P5)
```

Plus one bug fix discovered during training:
```python
# AMP fp16 vs LayerNorm fp32 dtype mismatch in the new index_copy_ path
out.index_copy_(1, active_idx, h.to(memory.dtype))
```

The full v2 module is at
[`rtdetr_pytorch/src/zoo/rtdetr/feedback_module.py`](rtdetr_pytorch/src/zoo/rtdetr/feedback_module.py)
(also mirrored in `v2_results/code/` for reproducibility).

---

## Repository layout

```
.
├── report/                         16-page LaTeX report (main.pdf is the deliverable)
│   ├── main.pdf                    ← final report
│   ├── main.tex
│   ├── sections/                   one .tex per section
│   ├── figures/                    plots + detection visualizations
│   ├── bibliography.bib
│   ├── Makefile                    `make` rebuilds main.pdf
│   └── scripts/generate_figures.py recreates plots from the JSON logs
│
├── v2_results/                     final v2 artifacts (positive result)
│   ├── ablations/                  3 ablation JSONs (ON 640, OFF 640, ON 800)
│   ├── logs/                       per-epoch log.txt + chain SLURM stdout
│   ├── code/                       exact code state at training time
│   └── README.md                   v2-specific notes
│
├── v1_results/                     v1 artifacts (negative-result baseline)
│   ├── ablations/                  4 v1 ablation JSONs
│   ├── logs/                       v1 per-epoch log.txt + chain stdout
│   ├── viz/                        5 detection visualizations
│   └── feedback_module.py          v1 module source for v1 vs v2 diff
│
├── rtdetr_pytorch/                 RT-DETR PyTorch source (modified)
│   ├── src/zoo/rtdetr/
│   │   ├── feedback_module.py      ← novel contribution (v2)
│   │   └── rtdetr_decoder.py       (kwargs threaded for feedback)
│   ├── configs/rtdetr/
│   │   └── rtdetr_r50vd_hpc_feedback_p2_v2.yml
│   └── tools/
│       ├── full_ablation.py        ON-vs-OFF ablation eval
│       ├── smoke_test_feedback.py  6 invariants for the feedback module
│       └── visualize_feedback.py
│
├── check_status.sh                 live HPC dashboard (Bocconi-internal use)
├── README_upstream.md              the original RT-DETR README (preserved)
└── README.md                       you are here
```

The trained checkpoints (`*.pth`, ~750 MB each) are excluded from git per
GitHub's 100 MB file-size limit. They live under
`v1_results/checkpoints/` and `v2_results/checkpoints/` on the author's local
machine; reach out if you need them.

---

## Reproducing the headline ablation

Given the v2 checkpoint locally, the +0.99 ΔAP\_S is reproduced by two calls
to `tools/full_ablation.py` against the same checkpoint:

```bash
cd rtdetr_pytorch

# feedback ON (default)
python tools/full_ablation.py \
    -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2_v2.yml \
    -r path/to/checkpoint_v2_final.pth \
    --label v2_on_640 --out-json /tmp/ablation_on.json

# feedback OFF (same checkpoint, only inference toggled)
python tools/full_ablation.py \
    -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2_v2.yml \
    -r path/to/checkpoint_v2_final.pth --feedback-off \
    --label v2_off_640 --out-json /tmp/ablation_off.json

# delta
python -c "
import json
on  = json.load(open('/tmp/ablation_on.json'))
off = json.load(open('/tmp/ablation_off.json'))
print(f'AP_S(ON) = {on[\"AP_S\"]*100:.2f}')
print(f'AP_S(OFF) = {off[\"AP_S\"]*100:.2f}')
print(f'Delta AP_S = {(on[\"AP_S\"]-off[\"AP_S\"])*100:+.2f}')
"
# expected output:
#   AP_S(ON) = 34.25
#   AP_S(OFF) = 33.26
#   Delta AP_S = +0.99
```

Pre-computed JSONs are in `v2_results/ablations/`.

---

## Training (for completeness)

The v2 checkpoint was produced by 13 effective epochs of finetuning from the
public `rtdetr_r50vd_6x_coco.pth` weights, on COCO `train2017` with
multi-scale [480..800], using AdamW with per-group learning rates (backbone
5e-7, all other groups 5e-5). Run as four chained SLURM jobs on a Bocconi
A100 80 GB (4g.40gb MIG slice), ~32 hours wall-clock.

The exact training script: `rtdetr_pytorch/scripts_hpc/train_p2_v2_chain.sbatch`.

---

## Rebuilding the report

The 16-page LaTeX report is in `report/`:

```bash
cd report
make           # regenerates figures + 3-pass pdflatex + bibtex
xdg-open main.pdf
```

Requires a working LaTeX install. The `Makefile` is configured for
`~/.TinyTeX/bin/x86_64-linux/pdflatex`; adjust `PDFLATEX` and `BIBTEX` paths
if your install is elsewhere. Figures are regenerated by
`scripts/generate_figures.py` from the JSONs and `log.txt` files in
`v1_results/` and `v2_results/`.

---

## Acknowledgments

- Built on top of [RT-DETR](https://github.com/lyuwenyu/RT-DETR) by Lyu et al.
  The original upstream README is preserved as `README_upstream.md`.
- Trained on the Bocconi University HPC cluster.
- Project pitch and progress feedback from co-author Nour Jennane.

## Authors

- **Hatem Saadallah** — Bocconi University, MSc in Artificial Intelligence
- **Nour Jennane** — Bocconi University, MSc in Artificial Intelligence

## License

Code modifications follow the upstream RT-DETR license (Apache-2.0).
The report and figures are released under CC-BY-4.0.
