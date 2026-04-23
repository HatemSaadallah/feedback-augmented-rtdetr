"""Convert xView GeoJSON annotations + TIFF images into a COCO-format
dataset (with tiling) that the RT-DETR pipeline can consume.

xView images are very large (~3000-4000 px), so we tile them into fixed-size
patches with overlap. Boxes are clipped per tile; tiles with no boxes are
dropped by default.

Layout produced under --out-dir:
    images/{split}/<image_id>_<ty>_<tx>.png
    annotations/xview_{split}.json       (COCO format)

Example:
    python tools/xview_to_coco.py \
        --geojson /path/to/xView_train.geojson \
        --images-dir /path/to/train_images \
        --out-dir data/xview_coco \
        --split train --tile 800 --overlap 0.2 --val-frac 0.1
"""

import argparse
import json
import os
import random
from collections import defaultdict

from PIL import Image

# xView class IDs are sparse (they go up to ~94 with gaps). We build a dense
# mapping from xView type_id → contiguous [0, K-1] category index on the fly.


def _load_xview_geojson(path: str):
    """Return dict: image_file_name -> list of (type_id, [x1,y1,x2,y2])."""
    with open(path) as f:
        gj = json.load(f)
    per_image = defaultdict(list)
    for feat in gj['features']:
        props = feat['properties']
        img = props.get('image_id') or props.get('IMAGE_ID')
        tid = int(props.get('type_id', props.get('TYPE_ID', -1)))
        # xView uses 'bounds_imcoords': "x1,y1,x2,y2"
        b = props.get('bounds_imcoords') or props.get('BOUNDS_IMCOORDS')
        if img is None or tid < 0 or b is None:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in b.split(',')]
        except ValueError:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        per_image[img].append((tid, [x1, y1, x2, y2]))
    return per_image


def _clip_box(box, x0, y0, x1, y1):
    bx1, by1, bx2, by2 = box
    ix1 = max(bx1, x0); iy1 = max(by1, y0)
    ix2 = min(bx2, x1); iy2 = min(by2, y1)
    if ix2 - ix1 <= 1 or iy2 - iy1 <= 1:
        return None
    # Keep a tile-local box only if at least half of the original is inside.
    orig_area = (bx2 - bx1) * (by2 - by1)
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    if inter_area < 0.3 * orig_area:
        return None
    return [ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0]


def _tiles(W: int, H: int, tile: int, overlap: float):
    stride = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < H: ys.append(max(0, H - tile))
    if not xs or xs[-1] + tile < W: xs.append(max(0, W - tile))
    for ty, y in enumerate(sorted(set(ys))):
        for tx, x in enumerate(sorted(set(xs))):
            yield tx, ty, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geojson', required=True)
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--split', default='train', choices=['train', 'trainval'])
    ap.add_argument('--tile', type=int, default=800)
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--val-frac', type=float, default=0.1,
                    help='If split=trainval, fraction of source images reserved as val.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-empty-tiles', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)
    per_image = _load_xview_geojson(args.geojson)
    all_images = sorted(per_image.keys())
    print(f'xView GeoJSON: {len(all_images)} images, '
          f'{sum(len(v) for v in per_image.values())} annotations')

    # Dense category map
    type_ids = sorted({tid for lst in per_image.values() for tid, _ in lst})
    cat_map = {tid: i for i, tid in enumerate(type_ids)}
    categories = [{'id': cat_map[tid], 'name': f'xview_{tid}', 'supercategory': 'object'}
                  for tid in type_ids]
    print(f'categories: {len(categories)}')

    # Split assignment
    split_of = {}
    if args.split == 'train':
        for n in all_images: split_of[n] = 'train'
    else:
        random.shuffle(all_images)
        cut = int(len(all_images) * (1 - args.val_frac))
        for n in all_images[:cut]: split_of[n] = 'train'
        for n in all_images[cut:]: split_of[n] = 'val'

    splits = defaultdict(lambda: {'images': [], 'annotations': [], 'categories': categories})
    tile_ids = defaultdict(int); ann_ids = defaultdict(int)

    for fname in all_images:
        split = split_of[fname]
        img_path = os.path.join(args.images_dir, fname)
        if not os.path.isfile(img_path):
            print(f'  skip (missing file): {fname}')
            continue
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            W, H = im.size
            for tx, ty, x0, y0 in _tiles(W, H, args.tile, args.overlap):
                x1, y1 = x0 + args.tile, y0 + args.tile
                tile_boxes = []
                for tid, box in per_image[fname]:
                    clipped = _clip_box(box, x0, y0, x1, y1)
                    if clipped is None: continue
                    tile_boxes.append((cat_map[tid], clipped))
                if not tile_boxes and not args.keep_empty_tiles:
                    continue
                # Save tile
                crop = im.crop((x0, y0, x1, y1))
                tid_str = os.path.splitext(fname)[0]
                out_name = f'{tid_str}_{ty}_{tx}.png'
                out_dir = os.path.join(args.out_dir, 'images', split)
                os.makedirs(out_dir, exist_ok=True)
                crop.save(os.path.join(out_dir, out_name), format='PNG')

                img_id = tile_ids[split]; tile_ids[split] += 1
                splits[split]['images'].append({
                    'id': img_id, 'file_name': out_name,
                    'width': args.tile, 'height': args.tile,
                })
                for cat_idx, (bx1, by1, bx2, by2) in tile_boxes:
                    aid = ann_ids[split]; ann_ids[split] += 1
                    w, h = bx2 - bx1, by2 - by1
                    splits[split]['annotations'].append({
                        'id': aid, 'image_id': img_id, 'category_id': cat_idx,
                        'bbox': [bx1, by1, w, h], 'area': float(w * h),
                        'iscrowd': 0, 'segmentation': [],
                    })

    ann_dir = os.path.join(args.out_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    for split, coco in splits.items():
        out_json = os.path.join(ann_dir, f'xview_{split}.json')
        with open(out_json, 'w') as f:
            json.dump(coco, f)
        print(f'{split}: {len(coco["images"])} tiles, '
              f'{len(coco["annotations"])} boxes → {out_json}')


if __name__ == '__main__':
    main()
