"""Laptop pipeline-validation: train feedback RT-DETR for a bounded number of
iterations on COCO val-as-train, logging loss + gate value + VRAM.

Not meant to produce a usable checkpoint — only to prove the training pipeline
works end-to-end on the local GPU (data, AMP, criterion, backward, feedback).

Usage:
    python tools/laptop_train_feedback.py \
        -c configs/rtdetr/rtdetr_r50vd_laptop_test_feedback.yml \
        --max-iters 500 --log-every 20 --amp
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.misc import dist


def _find_feedback(model):
    """Return the FeedbackAugmentedDecoder if present."""
    unwrapped = dist.de_parallel(model)
    transformer = getattr(unwrapped, 'decoder', None)
    if transformer is None:
        return None
    inner = getattr(transformer, 'decoder', None)
    if inner is None or not hasattr(inner, 'gate_value'):
        return None
    return inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-t', '--tuning', default=None,
                    help='Optional baseline checkpoint to warm-start from '
                         '(feedback params stay at init).')
    ap.add_argument('--max-iters', type=int, default=500)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = YAMLConfig(args.config, use_amp=args.amp)

    # Warm-start from a baseline checkpoint if requested.
    if args.tuning:
        state = torch.load(args.tuning, map_location='cpu')
        src_state = state['ema']['module'] if 'ema' in state else state['model']
        tgt_state = cfg.model.state_dict()
        matched = {k: v for k, v in src_state.items()
                   if k in tgt_state and v.shape == tgt_state[k].shape}
        missing = set(tgt_state) - set(matched)
        cfg.model.load_state_dict(matched, strict=False)
        print(f'tuning: loaded {len(matched)}/{len(tgt_state)} tensors from {args.tuning}')
        fb_missing = sum(1 for k in missing if 'feedback' in k)
        print(f'  feedback params not in baseline checkpoint: {fb_missing} '
              f'(expected — baseline has no feedback module)')

    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    optim = cfg.optimizer
    scaler = cfg.scaler

    loader = cfg.train_dataloader
    model.train(); criterion.train()

    feedback = _find_feedback(model)
    if feedback is None:
        print('WARNING: feedback module not found — running plain training.')
    else:
        feedback.set_feedback_disabled(False)
        print(f'feedback ready (initial gate = {feedback.gate_value:.5f})')

    t0 = time.perf_counter()
    step = 0
    running_loss = 0.0
    logged_losses = []
    gate_history = []

    for samples, targets in loader:
        if step >= args.max_iters:
            break
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type='cuda', cache_enabled=True):
                outputs = model(samples, targets)
            with torch.autocast(device_type='cuda', enabled=False):
                loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optim)
            scaler.update()
        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optim.step()

        running_loss += float(loss.item())
        step += 1

        if step % args.log_every == 0 or step == 1:
            avg = running_loss / max(1, min(step, args.log_every))
            gate_sig = feedback.gate_value if feedback else float('nan')
            gate_raw = feedback.feedback.gate.item() if feedback else float('nan')
            g_grad = feedback.feedback.gate.grad
            grad_val = float(g_grad.item()) if g_grad is not None else 0.0
            vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0
            dt = time.perf_counter() - t0
            ips = step / dt
            print(f'iter {step:4d}/{args.max_iters}  loss(avg{args.log_every})={avg:7.3f}  '
                  f'gate(raw)={gate_raw:+.8f}  gate(sig)={gate_sig:.7f}  '
                  f'gate.grad={grad_val:+.3e}  vram={vram:4.2f}GB  it/s={ips:4.2f}')
            logged_losses.append((step, avg))
            gate_history.append((step, gate_sig))
            running_loss = 0.0

    print('\n--- summary ---')
    print(f'iterations completed: {step}')
    if logged_losses:
        first, last = logged_losses[0][1], logged_losses[-1][1]
        print(f'loss: {first:.3f} → {last:.3f}  (delta {last - first:+.3f})')
    if feedback:
        init_gate = torch.sigmoid(torch.tensor(cfg.yaml_cfg.get('RTDETRTransformer', {}).get('feedback_gate_init', -6.0))).item()
        print(f'gate: {init_gate:.5f} (init) → {feedback.gate_value:.5f} (final)')
    if device.type == 'cuda':
        print(f'peak VRAM: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB')
    total = time.perf_counter() - t0
    print(f'total wall time: {total/60:.1f} min')


if __name__ == '__main__':
    main()
