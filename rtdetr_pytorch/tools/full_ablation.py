"""Comprehensive ablation eval for the trained P2-feedback RT-DETR.

Capabilities (per invocation):
  --feedback-off             disable the feedback module at inference
  --resolution H W           override eval input size (also overrides cached pos_embed)
  --latency-only             just measure latency / params, skip COCO eval
  --label NAME               row label for output JSON

Outputs a JSON (per row) with: AP, AP_S, AP_M, AP_L, AP50, AP75, AR, per_class_ap,
inference_ms_per_image, n_params.

Designed to be called repeatedly from a sbatch with different flags to build a full
results matrix without retraining.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset


COCO_CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush',
]


def _load_model(cfg_path, ckpt_path, feedback_off, resolution, device):
    cfg = YAMLConfig(cfg_path)
    if resolution is not None:
        # Patch eval_spatial_size on encoder + transformer + dataloader transforms
        H, W = resolution
        cfg.yaml_cfg.setdefault('HybridEncoder', {})['eval_spatial_size'] = [H, W]
        cfg.yaml_cfg.setdefault('RTDETRTransformer', {})['eval_spatial_size'] = [H, W]
        # Override val_dataloader transforms to resize at the new size
        if 'val_dataloader' in cfg.yaml_cfg:
            cfg.yaml_cfg['val_dataloader']['dataset']['transforms'] = {
                'type': 'Compose',
                'ops': [
                    {'type': 'Resize', 'size': [H, W]},
                    {'type': 'ToImageTensor'},
                    {'type': 'ConvertDtype'},
                ],
            }

    state = torch.load(ckpt_path, map_location='cpu')
    src_state = state['ema']['module'] if 'ema' in state else state['model']
    tgt_state = cfg.model.state_dict()
    matched = {k: v for k, v in src_state.items() if k in tgt_state and v.shape == tgt_state[k].shape}
    cfg.model.load_state_dict(matched, strict=False)

    model = cfg.model.to(device).eval()
    postproc = cfg.postprocessor.eval()
    criterion = cfg.criterion.to(device)

    # Disable feedback at inference if requested
    if feedback_off:
        try:
            inner = getattr(model.decoder, 'decoder', None)
            if inner is not None and hasattr(inner, 'set_feedback_disabled'):
                inner.set_feedback_disabled(True)
                print('feedback DISABLED at inference')
            else:
                print('warning: model has no FeedbackAugmentedDecoder wrapper')
        except Exception as e:
            print(f'warning: could not disable feedback: {e}')
    return cfg, model, postproc, criterion


def _eval_coco(cfg, model, postproc, criterion, device):
    from src.solver.det_engine import evaluate
    val_loader = cfg.val_dataloader
    base_ds = get_coco_api_from_dataset(val_loader.dataset)
    stats, coco_eval = evaluate(model, criterion, postproc, val_loader, base_ds, device, cfg.output_dir)
    return stats, coco_eval


def _per_class_ap(coco_eval):
    """Extract per-class AP for the bbox iouType."""
    ce = coco_eval.coco_eval.get('bbox')
    if ce is None:
        return {}
    # ce.eval['precision'] shape: [TxRxKxAxM] = [iouThrs, recThrs, K=cat, A=area, M=maxDets]
    p = ce.eval.get('precision')
    if p is None:
        return {}
    # mean over IoU thresholds, recall thresholds, all area, maxDets=100 (idx -1)
    ap_per_cat = []
    for k in range(p.shape[2]):
        sub = p[:, :, k, 0, -1]
        sub = sub[sub > -1]
        ap_per_cat.append(float(sub.mean()) if sub.size else float('nan'))

    # Map COCO category IDs to names
    cat_ids = sorted(ce.cocoGt.getCatIds())
    out = {}
    for i, cid in enumerate(cat_ids):
        name = ce.cocoGt.loadCats([cid])[0]['name']
        if i < len(ap_per_cat):
            out[name] = ap_per_cat[i]
    return out


def _latency(model, device, input_size=640, n_warmup=20, n_iters=200):
    H = W = input_size
    dummy = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 1000.0 * dt / n_iters  # ms per image at batch=1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--ckpt', required=True)
    ap.add_argument('--feedback-off', action='store_true')
    ap.add_argument('--resolution', type=int, nargs=2, default=None,
                    help='override eval input size as H W (e.g. 480 480)')
    ap.add_argument('--latency-only', action='store_true')
    ap.add_argument('--latency-input-size', type=int, default=640)
    ap.add_argument('--label', default='ablation')
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'=== full_ablation: label={args.label} ===')
    print(f'config={args.config}, ckpt={args.ckpt}')
    print(f'feedback_off={args.feedback_off}, resolution={args.resolution}, latency_only={args.latency_only}')

    cfg, model, postproc, criterion = _load_model(
        args.config, args.ckpt, args.feedback_off, args.resolution, device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f'n_params: {n_params / 1e6:.2f} M')

    out = {
        'label': args.label,
        'config': args.config,
        'ckpt': args.ckpt,
        'feedback_off': args.feedback_off,
        'resolution': args.resolution,
        'n_params': n_params,
    }

    if args.latency_only:
        print('--- latency benchmark ---')
        ms = _latency(model, device, input_size=args.latency_input_size)
        out['inference_ms_per_image'] = ms
        out['fps'] = 1000.0 / ms
        print(f'inference: {ms:.2f} ms/img  ({1000.0/ms:.1f} FPS)')
    else:
        print('--- COCO eval ---')
        stats, coco_eval = _eval_coco(cfg, model, postproc, criterion, device)
        ap_arr = stats.get('coco_eval_bbox', [None] * 12)
        out.update({
            'AP':    ap_arr[0],
            'AP50':  ap_arr[1],
            'AP75':  ap_arr[2],
            'AP_S':  ap_arr[3],
            'AP_M':  ap_arr[4],
            'AP_L':  ap_arr[5],
            'AR_1':  ap_arr[6],
            'AR_10': ap_arr[7],
            'AR_100':ap_arr[8],
            'AR_S':  ap_arr[9],
            'AR_M':  ap_arr[10],
            'AR_L':  ap_arr[11],
        })
        try:
            out['per_class_ap'] = _per_class_ap(coco_eval)
        except Exception as e:
            print(f'per-class AP extraction failed: {e}')
            out['per_class_ap'] = {}

    if args.out_json is None:
        args.out_json = f'/home/3415496/rt-detr/ablation_{args.label}.json'
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nsaved: {args.out_json}')

    print('\n=== summary ===')
    for k in ['AP', 'AP_S', 'AP_M', 'AP_L', 'inference_ms_per_image', 'fps', 'n_params']:
        if k in out:
            print(f'  {k:<25} {out[k]}')


if __name__ == '__main__':
    main()
