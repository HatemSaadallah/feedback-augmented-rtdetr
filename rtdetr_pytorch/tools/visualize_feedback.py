"""Visualize what the decoder-to-encoder feedback is attending to.

Produces a 1x3 panel for a given image:
    [ baseline detections | feedback detections | feedback attention heatmap ]

The attention heatmap aggregates feedback attention weights across the 300
decoder queries, giving a per-S5-token score that is upsampled to image size
and overlaid.

Example:
    python tools/visualize_feedback.py \
        --baseline-config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        --baseline-ckpt   output/rtdetr_r50vd_6x_coco/checkpoint.pth \
        --feedback-config configs/rtdetr/rtdetr_r50vd_6x_coco_feedback.yml \
        --feedback-ckpt   output/rtdetr_r50vd_6x_coco_feedback/checkpoint.pth \
        --image path/to/image.jpg --out viz.png --score-thr 0.5
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


INPUT_SIZE = 640


def _load_model(config_path: str, ckpt_path: str, device) -> nn.Module:
    cfg = YAMLConfig(config_path)
    state = torch.load(ckpt_path, map_location='cpu')
    model_state = state['ema']['module'] if 'ema' in state else state['model']
    cfg.model.load_state_dict(model_state, strict=False)
    model = cfg.model.to(device).eval()
    postproc = cfg.postprocessor.to(device).eval()
    return model, postproc


def _preprocess(image_path: str, device) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    im = Image.open(image_path).convert('RGB')
    w, h = im.size
    tr = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])
    x = tr(im).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]], device=device)
    return im, x, orig_size


@torch.no_grad()
def _run(model, postproc, x: torch.Tensor, orig_size: torch.Tensor):
    out = model(x)
    dets = postproc(out, orig_size)[0]
    return out, dets


def _s5_attention_heatmap(model) -> np.ndarray:
    """Return a 2D heatmap [H5, W5] of per-S5-token feedback strength.

    Uses the most recent attention weights stashed by the feedback module:
    shape [B, H5*W5, num_queries]. We take the max over queries per token so
    each spatial location reflects "the strongest decoder guidance it received".
    """
    fb = model.decoder.decoder.feedback  # RTDETR -> RTDETRTransformer -> FeedbackAugmentedDecoder -> feedback
    aw = fb.last_attn_weights  # [B, L, Q]
    if aw is None:
        raise RuntimeError('Feedback attention weights not stashed — was feedback disabled?')
    per_token = aw[0].max(dim=-1).values  # [L]
    n = per_token.numel()
    h = w = int(round(n ** 0.5))
    assert h * w == n, f'S5 slice {n} is not a perfect square'
    return per_token.reshape(h, w).detach().cpu().numpy()


def _draw_detections(ax, im: Image.Image, dets: dict, title: str, score_thr: float):
    ax.imshow(im)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    boxes  = dets['boxes'].cpu().numpy()
    labels = dets['labels'].cpu().numpy()
    scores = dets['scores'].cpu().numpy()
    keep = scores > score_thr
    for (x1, y1, x2, y2), lab, sc in zip(boxes[keep], labels[keep], scores[keep]):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1.5, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 2), f'{int(lab)}:{sc:.2f}',
                fontsize=7, color='black',
                bbox=dict(facecolor='lime', edgecolor='none', pad=1))


def _draw_heatmap(ax, im: Image.Image, heat: np.ndarray, title: str):
    W, H = im.size
    heat_resized = np.array(Image.fromarray(heat).resize((W, H), Image.BILINEAR))
    hr = heat_resized
    hr = (hr - hr.min()) / max(hr.max() - hr.min(), 1e-9)
    ax.imshow(im)
    ax.imshow(hr, cmap='jet', alpha=0.45)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-config', required=True)
    ap.add_argument('--baseline-ckpt',   required=True)
    ap.add_argument('--feedback-config', required=True)
    ap.add_argument('--feedback-ckpt',   required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', default='feedback_viz.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--score-thr', type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(args.device)
    im, x, orig = _preprocess(args.image, device)

    base_model, base_post = _load_model(args.baseline_config, args.baseline_ckpt, device)
    feed_model, feed_post = _load_model(args.feedback_config, args.feedback_ckpt, device)

    # Ensure feedback is engaged at eval time regardless of last training state.
    feed_model.decoder.decoder.set_feedback_disabled(False)

    _, base_dets = _run(base_model, base_post, x, orig)
    _, feed_dets = _run(feed_model, feed_post, x, orig)
    heat = _s5_attention_heatmap(feed_model)

    gate = feed_model.decoder.decoder.gate_value
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _draw_detections(axes[0], im, base_dets, 'Baseline', args.score_thr)
    _draw_detections(axes[1], im, feed_dets, f'Feedback (gate={gate:.3f})', args.score_thr)
    _draw_heatmap(axes[2], im, heat, 'S5 feedback attention')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'saved → {args.out}')


if __name__ == '__main__':
    main()
