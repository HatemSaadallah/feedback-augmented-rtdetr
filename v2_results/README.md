# RT-DETR with Decoder-to-Encoder Feedback — v2 Results

Local backup of the v2 retrain that turned the original negative result (v1, ΔAP_S = 0.0) into a measurable causal contribution from the feedback mechanism (v2, **ΔAP_S = +0.99**).

## Headline numbers (COCO val2017)

```
                       AP        AP_S       AP_M       AP_L
v2 ON  @ 640         52.10      34.25      56.33      68.82
v2 OFF @ 640         51.53      33.26      55.91      68.32
                     ─────      ─────      ─────      ─────
Δ (causal)           +0.57      +0.99      +0.42      +0.50

v2 ON  @ 800         52.31      37.01      56.35      67.04
```

ΔAP_S = AP_S(feedback ON) − AP_S(feedback OFF) on the same final checkpoint, only inference-time behavior changed (feedback module bypassed via `set_feedback_disabled(True)`).

## v1 vs v2

| Metric         | v1 ON  | v1 OFF | v1 Δ  | v2 ON  | v2 OFF | v2 Δ   |
|----------------|--------|--------|-------|--------|--------|--------|
| AP             | 52.01  | ~52.0  | ~0.00 | 52.10  | 51.53  | +0.57  |
| AP_S           | 33.91  | 33.91  |  0.00 | 34.25  | 33.26  | **+0.99** |
| AP_S @ 800     | 36.74  |   —    |   —   | 37.01  |   —    |   —    |

v1 was statistically silent at inference because the gate decayed to ~0.12 during training. v2 holds the gate near 0.39 via reparameterization, so the trained cross-attention actually flows into memory at inference.

## v1 → v2 code changes

Three changes in `code/feedback_module.py` (and corresponding kwargs in `rtdetr_decoder.py`):

1. **`gate_init`: -2.0 → 0.0.** σ(0)=0.5 vs σ(-2)≈0.12; feedback is meaningful from step 1.
2. **`gate_floor` (NEW): 0.1 with reparameterization.** Effective gate is
   `gate_eff = floor + (1 − floor) · σ(α)`; gate can never collapse below `floor`,
   regardless of optimizer pressure.
3. **`level_mask` (NEW): `[T, T, F, F]`.** Cross-attention runs on a subset of
   the encoder memory (P2 + P3 only), preserving S4/S5 byte-identical via
   `index_copy_`. Targets the levels where small-object information lives.

Net change in parameters: **0** (no new params added; only reparameterization + masked compute).

There was also a dtype bug fix: `out.index_copy_(1, active_idx, h.to(memory.dtype))` to handle AMP fp16 vs LayerNorm fp32 mismatch.

## Training facts

- 13 epochs total across 4 SLURM chain links (1 + 4 + 4 + 4)
- Per-link wall: ~2h 30m × 4 = ~10h, total ~32h compute on Bocconi `stud` partition (A100 MIG slice)
- LR: backbone 5e-7, feedback / encoder-norm / decoder-norm / catchall 5e-5; no decay (milestones[9] never triggered within a 4-epoch chain link, since `--tuning` resets the scheduler each link)
- 13 epochs because chain link 1 crashed at start of epoch 1 due to the AMP dtype bug; subsequent links each ran their full 4

## Per-epoch AP_S trajectory (with feedback ON)

```
ep  0: 23.55   chain link 1 (single epoch)
ep  1: 24.00   chain link 2 fresh start (--tuning resets optimizer momentum)
ep  2: 31.55   momentum recovered, +7.5 jump
ep  3: 32.16
ep  4: 33.05
ep  5: 33.00
ep  6: 33.27
ep  7: 33.27
ep  8: 34.05   first to break v1's 33.91 final
ep  9: 33.78
ep 10: 33.79
ep 11: 34.31
ep 12: 34.30   ← FINAL
```

## Gate trajectory

```
init   0.55
ep  4  0.4676
ep  5  0.4494
ep  6  0.4356
ep  7  0.4356  flat
ep  8  0.4186
ep 11  0.3871
ep 12  0.39    ≈ steady-state
floor  0.10    never approached
```

## Directory contents

```
checkpoints/
  checkpoint_v2_final.pth         epoch 12 final state, 49.08 M params

ablations/
  ablation_v2_feedback_on_640.json   AP, AP_S, per-class AP, n_params
  ablation_v2_feedback_off_640.json  same, with feedback bypassed at inference
  ablation_v2_feedback_on_800.json   robustness check at higher resolution

logs/
  log.txt                            13 JSON entries, one per epoch
  eval_p2_v2_486313.out              full ablation eval stdout (incl. summary line)
  p2_v2_chain_484637.out             chain link 1 (crashed at dtype bug)
  p2_v2_chain_484871.out             chain link 2 (epochs 1-4)
  p2_v2_chain_485450.out             chain link 3 (epochs 5-8)
  p2_v2_chain_485950.out             chain link 4 (epochs 9-12 + final eval)

code/
  feedback_module.py                 the v2 module (gate floor, level mask, dtype fix)
  rtdetr_decoder.py                  v2 kwargs threaded through RTDETRTransformer
  rtdetr_r50vd_hpc_feedback_p2_v2.yml   exact config used for training
  train_p2_v2_chain.sbatch           SLURM training chain script
  eval_all_p2_v2.sbatch              SLURM ablation eval script
```

## Reproducing the eval locally

The exact ablation can be re-run from `checkpoints/checkpoint_v2_final.pth` using the v2 code in `code/`:

```bash
cd <RT-DETR repo>/rtdetr_pytorch
# replace the in-tree feedback_module.py and rtdetr_decoder.py with v2_results/code/*
# replace configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2_v2.yml with v2_results/code/*

python tools/full_ablation.py \
    -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2_v2.yml \
    -r path/to/checkpoint_v2_final.pth \
    --label v2_on_640 --out-json /tmp/ablation_on.json

python tools/full_ablation.py \
    -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2_v2.yml \
    -r path/to/checkpoint_v2_final.pth --feedback-off \
    --label v2_off_640 --out-json /tmp/ablation_off.json
```

The two JSONs reproduce `ON @ 640` and `OFF @ 640` rows of the headline table.

## Caveats and honesty

- v2's absolute AP_S of 34.25 is below the RT-DETR paper baseline of 34.7 (a 13-epoch finetune from baseline weights vs the paper's longer schedule). The contribution claim is the **+0.99 ΔAP_S from feedback on the same checkpoint**, not "best AP_S on COCO."
- v2's latency is ~25 FPS @ 640 vs the baseline's ~53 FPS — roughly 2× slower. The added P2 stride-4 level dominates this cost; the feedback module itself is small.
- Parameter count is +14% over RT-DETR baseline (49.08M vs 42.9M), again from the P2 level rather than feedback (which adds 0 net params in v2 since it's just reparameterization + masked attention).
- No multi-seed run; the +0.99 is a single-seed comparison. Replicating across seeds would strengthen the claim.

## v1 reference

For the negative-result baseline, see `/home/hatem/Desktop/rt-detr/v1_results/` (separate directory, similar structure).
