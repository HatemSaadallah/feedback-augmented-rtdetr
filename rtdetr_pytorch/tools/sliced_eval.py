"""Sliced inference (SAHI-style) eval on COCO val for RT-DETR.

For each val image:
    1. If max(H, W) > slice_size: tile into overlapping slice_size x slice_size patches.
    2. Run the model on each patch.
    3. Map predicted boxes back to the original image coordinates.
    4. Run class-aware NMS to merge across patches.

Output: COCO-style predictions JSON + standard pycocotools AP / AP_S / AP_M / AP_L.

Usage:
    python tools/sliced_eval.py \
        -c configs/rtdetr/rtdetr_r50vd_hpc_feedback_p2.yml \
        -r /home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2/checkpoint.pth \
        --slice-size 640 --overlap 0.2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_model(config_path: str, ckpt_path: str, device):
    cfg = YAMLConfig(config_path)
    state = torch.load(ckpt_path, map_location='cpu')
    model_state = state['ema']['module'] if 'ema' in state else state['model']
    cfg.model.load_state_dict(model_state, strict=False)
    model = cfg.model.to(device).eval()
    postproc = cfg.postprocessor.to(device).eval()
    num_top = getattr(cfg.postprocessor, 'num_top_queries', 300)
    return model, postproc, num_top


def get_val_paths(yaml_path: str):
    """Pull img_folder + ann_file from the model config's val_dataloader."""
    cfg = YAMLConfig(yaml_path)
    yaml = cfg.yaml_cfg
    val = yaml.get('val_dataloader', {}).get('dataset', {})
    return val.get('img_folder'), val.get('ann_file')


def slice_image(W: int, H: int, slice_size: int, overlap: float):
    """Yield (x0, y0, x1, y1) windows tiling the image with overlap."""
    if W <= slice_size and H <= slice_size:
        yield 0, 0, W, H
        return
    stride = max(1, int(slice_size * (1 - overlap)))
    xs = list(range(0, max(W - slice_size, 0) + 1, stride))
    ys = list(range(0, max(H - slice_size, 0) + 1, stride))
    if not xs or xs[-1] + slice_size < W: xs.append(max(0, W - slice_size))
    if not ys or ys[-1] + slice_size < H: ys.append(max(0, H - slice_size))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            yield x, y, min(x + slice_size, W), min(y + slice_size, H)


@torch.no_grad()
def predict_slice(model, postproc, slice_pil, device, model_input_size: int = 640):
    """Run model on a single slice. Returns (boxes_xyxy_in_slice, scores, labels)."""
    sw, sh = slice_pil.size
    tr = T.Compose([T.Resize((model_input_size, model_input_size)), T.ToTensor()])
    x = tr(slice_pil).unsqueeze(0).to(device)
    out = model(x)
    orig = torch.tensor([[sw, sh]], device=device)
    res = postproc(out, orig)[0]
    return res['boxes'].cpu().numpy(), res['scores'].cpu().numpy(), res['labels'].cpu().numpy()


def per_class_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                  iou_thr: float = 0.6) -> np.ndarray:
    """Class-aware NMS. Returns indices to keep."""
    keep_all = []
    for c in np.unique(labels):
        m = labels == c
        if not m.any():
            continue
        b_c = torch.from_numpy(boxes[m]).float()
        s_c = torch.from_numpy(scores[m]).float()
        idx = torchvision.ops.nms(b_c, s_c, iou_thr).numpy()
        # Map back to global indices
        global_idx = np.flatnonzero(m)[idx]
        keep_all.append(global_idx)
    if not keep_all:
        return np.array([], dtype=int)
    return np.concatenate(keep_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--ckpt', required=True)
    ap.add_argument('--slice-size', type=int, default=640)
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--score-thr', type=float, default=0.05)
    ap.add_argument('--nms-iou', type=float, default=0.6)
    ap.add_argument('--out-json', default='/tmp/sliced_preds.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='for debugging, max images')
    args = ap.parse_args()

    device = torch.device(args.device)
    img_dir, ann_file = get_val_paths(args.config)
    print(f'val images: {img_dir}')
    print(f'val annotations: {ann_file}')

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    if args.limit:
        img_ids = img_ids[:args.limit]
    print(f'evaluating on {len(img_ids)} images')

    model, postproc, num_top = load_model(args.config, args.ckpt, device)

    # Build cat_id → contiguous mapping. Our model emits contiguous 0..79;
    # COCO eval needs original cat_ids 1..90 with gaps.
    from src.data.coco.coco_dataset import mscoco_label2category
    cat_id_map = {i: mscoco_label2category[i] for i in range(80)}

    coco_preds = []
    t0 = time.perf_counter()
    for i, iid in enumerate(img_ids):
        info = coco_gt.loadImgs([iid])[0]
        path = os.path.join(img_dir, info['file_name'])
        try:
            pil = Image.open(path).convert('RGB')
        except Exception as e:
            print(f'[{i}] skip {path}: {e}'); continue
        W, H = pil.size

        all_boxes, all_scores, all_labels = [], [], []
        for x0, y0, x1, y1 in slice_image(W, H, args.slice_size, args.overlap):
            slice_pil = pil.crop((x0, y0, x1, y1))
            sw, sh = x1 - x0, y1 - y0
            boxes, scores, labels = predict_slice(model, postproc, slice_pil, device, args.slice_size)
            # boxes in slice coordinates; offset to image coordinates
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            mask = scores > args.score_thr
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
            all_labels.append(labels[mask])
        if not all_boxes:
            continue
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4))
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        if len(boxes) == 0:
            continue
        keep = per_class_nms(boxes, scores, labels, args.nms_iou)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Cap at num_top_queries to match standard eval
        if len(boxes) > num_top:
            top = np.argsort(-scores)[:num_top]
            boxes, scores, labels = boxes[top], scores[top], labels[top]

        for box, sc, lab in zip(boxes, scores, labels):
            cat_id = cat_id_map.get(int(lab))
            if cat_id is None:
                # Defensive: if model emitted a class index outside the 0..79 contiguous
                # range (e.g. denoising padding label), skip this prediction.
                continue
            x1b, y1b, x2b, y2b = box.tolist()
            coco_preds.append({
                'image_id': iid,
                'category_id': cat_id,
                'bbox': [x1b, y1b, x2b - x1b, y2b - y1b],
                'score': float(sc),
            })
        if (i + 1) % 200 == 0:
            dt = time.perf_counter() - t0
            print(f'  [{i+1}/{len(img_ids)}] elapsed={dt/60:.1f}min  preds={len(coco_preds)}')

    print(f'\ntotal predictions: {len(coco_preds)}')
    with open(args.out_json, 'w') as f:
        json.dump(coco_preds, f)
    print(f'wrote {args.out_json}')

    # COCO eval
    coco_dt = coco_gt.loadRes(args.out_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
