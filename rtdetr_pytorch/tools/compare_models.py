"""Evaluate baseline and feedback RT-DETR checkpoints on COCO val and compare.

Reports per-checkpoint COCO AP/AP_50/AP_75/AP_S/AP_M/AP_L plus measured FPS,
then prints a side-by-side table.

Usage:
    python tools/compare_models.py \
        --baseline-config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        --baseline-ckpt   output/rtdetr_r50vd_6x_coco/checkpoint.pth \
        --feedback-config configs/rtdetr/rtdetr_r50vd_6x_coco_feedback.yml \
        --feedback-ckpt   output/rtdetr_r50vd_6x_coco_feedback/checkpoint.pth \
        --device cuda --fps-iters 200
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset
from src.solver.det_engine import evaluate


COCO_STAT_NAMES = [
    'AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L',
    'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L',
]


def _load_checkpoint_into(cfg, ckpt_path: str):
    state = torch.load(ckpt_path, map_location='cpu')
    model_state = state['ema']['module'] if 'ema' in state else state['model']
    cfg.model.load_state_dict(model_state, strict=False)


def _evaluate_coco(cfg, device):
    model = cfg.model.to(device).eval()
    postproc = cfg.postprocessor
    criterion = cfg.criterion
    val_loader = cfg.val_dataloader
    base_ds = get_coco_api_from_dataset(val_loader.dataset)
    stats, _ = evaluate(model, criterion, postproc, val_loader, base_ds, device, cfg.output_dir)
    coco_stats = stats['coco_eval_bbox']
    return dict(zip(COCO_STAT_NAMES, coco_stats))


def _measure_fps(cfg, device, iters: int = 200, input_size: int = 640) -> float:
    model = cfg.model.to(device).eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(20):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return iters / dt


def _run_one(name: str, config: str, ckpt: str, device, fps_iters: int) -> dict:
    print(f'\n========== {name} ==========')
    cfg = YAMLConfig(config)
    _load_checkpoint_into(cfg, ckpt)
    coco = _evaluate_coco(cfg, device)
    fps = _measure_fps(cfg, device, iters=fps_iters)
    print(f'[{name}] FPS = {fps:.2f}')
    return {'name': name, 'coco': coco, 'fps': fps}


def _print_comparison(base: dict, feed: dict):
    print('\n' + '=' * 62)
    print(f'{"metric":<10} {"baseline":>12} {"feedback":>12} {"delta":>12}')
    print('-' * 62)
    for k in ['AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L']:
        b = base['coco'][k]
        f = feed['coco'][k]
        print(f'{k:<10} {b*100:>11.2f}% {f*100:>11.2f}% {(f-b)*100:>+11.2f}')
    print(f'{"FPS":<10} {base["fps"]:>12.2f} {feed["fps"]:>12.2f} '
          f'{feed["fps"]-base["fps"]:>+12.2f}')
    print('=' * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-config', required=True)
    ap.add_argument('--baseline-ckpt', required=True)
    ap.add_argument('--feedback-config', required=True)
    ap.add_argument('--feedback-ckpt', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fps-iters', type=int, default=200)
    args = ap.parse_args()

    device = torch.device(args.device)
    base = _run_one('baseline', args.baseline_config, args.baseline_ckpt, device, args.fps_iters)
    feed = _run_one('feedback', args.feedback_config, args.feedback_ckpt, device, args.fps_iters)
    _print_comparison(base, feed)


if __name__ == '__main__':
    main()
