"""Score a checkpoint on held-out frames with the organisers' matcher: threshold sweep, per-folder table, benchmark json.

    uv run python training/eval_ckpt.py --weights runs/exp/NAME/weights/best.pt --images work/yolo_v/val_images \
        --ann ~/data/train/annotations.xml --out work/res_NAME.json

Leaves weights/ untouched. Prints, for each threshold, the leaderboard objective (fps term excluded) overall and per
folder, then writes a benchmark-format json at the best overall threshold (feed it to training/per_folder.py).
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

from surgnet.inference import SurgNet
from surgnet.labelio import load_cvat_xml
from surgnet.scoring import match_tips, compute_metrics
from surgnet.shared.images import find_images, read_image

THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]


def objective(items):
    o = compute_metrics(items)['overall']
    err = o['median_error_px']
    return 0.4 * o['precision'] + 0.35 * o['recall'] + 0.15 * (0 if err is None else max(0.0, 1 - err / 50)), o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--imgsz', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    gt = load_cvat_xml(a.ann)
    names = [n for n in find_images(a.images) if n in gt]
    model = SurgNet(weights=a.weights)
    if a.imgsz:
        model.cfg['imgsz'] = a.imgsz
    model.cfg['conf_by_class'] = None
    raw = {n: model.predict_raw(read_image(Path(a.images) / n), conf=0.05) for n in tqdm(names, desc='predict')}
    folders = sorted({n.split('/')[0] for n in names})

    def matched(thr, subset):
        out = []
        for n in subset:
            g = gt[n]
            diag = float(np.hypot(g.width, g.height))
            out.append(match_tips(model._dedupe([d for d in raw[n] if d.confidence >= thr], diag), g.tips, 0.025 * diag))
        return out

    print(f'{"conf":>5} {"ALL":>6} {"P":>5} {"R":>5} {"err":>5}  ' + '  '.join(f'{f[:10]:>10}' for f in folders))
    best = None
    for thr in THRESHOLDS:
        obj, o = objective(matched(thr, names))
        per = [objective(matched(thr, [n for n in names if n.startswith(f + '/')]))[0] for f in folders]
        print(f'{thr:5.2f} {obj:6.3f} {o["precision"]:5.2f} {o["recall"]:5.2f} {o["median_error_px"] or 0:5.1f}  '
              + '  '.join(f'{p:10.3f}' for p in per))
        if best is None or obj > best[0]:
            best = (obj, thr)
    thr = best[1]
    for frac in (0.02, 0.04):                     # does merging different-class tips on one spot help precision?
        model.cfg['cross_class_frac'] = frac
        obj, o = objective(matched(thr, names))
        print(f'[eval] conf {thr} + cross-class dedupe {frac}: objective {obj:.3f} P {o["precision"]:.3f} R {o["recall"]:.3f}')
    model.cfg['cross_class_frac'] = 0.0
    m = compute_metrics(matched(thr, names))
    print(f'[eval] best conf {thr}: objective {best[0]:.3f} (+0.10 if fps >= 100)  per class F1: '
          + '  '.join(f'{c}={s["f1"]:.2f}' for c, s in m['per_class'].items()))
    if a.out:
        images = {n: {'latency_ms': None, 'error': None,
                      'detections': [d.__dict__ | {'tip_xy': list(d.tip_xy)} for d in raw[n] if d.confidence >= thr]}
                  for n in names}
        Path(a.out).write_text(json.dumps({'images': images, 'latency': {'fps': 100.0}, 'conf': thr}))


if __name__ == '__main__':
    main()
