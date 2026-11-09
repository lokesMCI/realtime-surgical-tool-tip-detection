"""Overlay ground-truth tips (green) and model predictions (red) on frames, as a contact sheet per folder.

    uv run python training/viz.py --images work/yolo/val_images --ann ~/data/train/annotations.xml --n 8 [--pred]

Look at it before trusting any number: do the green dots sit on the tool tips? where do the red ones go wrong?
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import cv2
import numpy as np

from surgnet.labelio import load_cvat_xml
from surgnet.shared.images import find_images, read_image


def draw(img, gt_tips, preds):
    for t in gt_tips:
        p = (int(t.tip_xy[0]), int(t.tip_xy[1]))
        cv2.circle(img, p, 14, (0, 255, 0), 3)
        cv2.putText(img, t.class_name, (p[0] + 16, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for d in preds:
        p = (int(d.tip_xy[0]), int(d.tip_xy[1]))
        cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.putText(img, f'{d.class_name} {d.confidence:.2f}', (p[0] + 16, p[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='work/yolo/val_images')
    ap.add_argument('--ann', default='data/annotations.xml')
    ap.add_argument('--out', default='work/viz')
    ap.add_argument('--n', type=int, default=8, help='frames per folder')
    ap.add_argument('--pred', action='store_true', help='also run the shipped model')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    gt = load_cvat_xml(a.ann)
    names = [n for n in find_images(a.images) if n in gt]
    model = None
    if a.pred:
        from surgnet.inference import SurgNet
        model = SurgNet()
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for folder in sorted({n.split('/')[0] for n in names}):
        pick = random.sample([n for n in names if n.startswith(folder + '/')], k=min(a.n, sum(n.startswith(folder + '/') for n in names)))
        tiles = []
        for n in pick:
            img = read_image(Path(a.images) / n)
            tiles.append(cv2.resize(draw(img, gt[n].tips, model.predict(img) if model else []), (480, 540)))
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(out / f'{folder}.jpg'), sheet)
        print('wrote', out / f'{folder}.jpg')


if __name__ == '__main__':
    main()
