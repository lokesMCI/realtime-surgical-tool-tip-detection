"""Pick the confidence threshold on the by-clip val frames with the organiser's own matcher, and write it to config.json.

    uv run python training/tune_threshold.py --images work/yolo/val_images --ann data/annotations.xml

The model runs once at a very low threshold; every candidate threshold is then applied to those cached detections, so the
sweep costs nothing. The objective is the leaderboard formula without the fps term (fps does not depend on the threshold):
0.40 precision + 0.35 recall + 0.15 (1 - median error / 50). Frozen into weights/config.json; do not re-tune on the leaderboard.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

# Avoid PyTorch's shared-memory unlink race while loading/predicting across processes.
try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    # The strategy may already be configured once PyTorch multiprocessing has started.
    pass

from surgnet.inference import SurgNet, WEIGHTS_DIR
from surgnet.labelio import load_cvat_xml
from surgnet.scoring import match_tips, compute_metrics
from surgnet.shared.images import find_images, read_image
from surgnet.shared.detection import ToolDetection


def score_at(raw, gt, names, thr, model, radius_frac=0.025):
    per_image = []
    for n in names:
        g = gt[n]
        diag = float(np.hypot(g.width, g.height))
        dets = [d for d in raw[n] if d.confidence >= thr]
        dets = model._dedupe(dets, diag)
        per_image.append(match_tips(dets, g.tips, radius_frac * diag))
    m = compute_metrics(per_image)['overall']
    err = m['median_error_px']
    err_c = 0.0 if err is None else min(1.0, max(0.0, 1 - err / 50.0))
    obj = 0.40 * m['precision'] + 0.35 * m['recall'] + 0.15 * err_c
    return obj, m, err_c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='work/yolo/val_images')
    ap.add_argument('--ann', default='data/annotations.xml')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--no-write', action='store_true')
    a = ap.parse_args()

    gt = load_cvat_xml(a.ann)
    names = [n for n in find_images(a.images) if n in gt]
    if not names:
        raise SystemExit('no val image has labels: check --images/--ann (names must match)')
    model = SurgNet(device=a.device, weights=a.weights)
    raw = {n: model.predict_raw(read_image(Path(a.images) / n), conf=0.001) for n in tqdm(names, desc='predict val')}

    best = None
    print(f'{"conf":>6} {"obj":>6} {"prec":>6} {"recall":>6} {"err_px":>7} {"tp":>5} {"fp":>5} {"fn":>5}')
    for thr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        obj, m, _ = score_at(raw, gt, names, thr, model)
        print(f'{thr:6.2f} {obj:6.3f} {m["precision"]:6.3f} {m["recall"]:6.3f} {m["median_error_px"] or float("nan"):7.1f} '
              f'{m["tp"]:5d} {m["fp"]:5d} {m["fn"]:5d}')
        if best is None or obj > best[0]:
            best = (obj, thr)
    print(f'[tune] best conf = {best[1]} (objective {best[0]:.3f} on {len(names)} val frames)')
    if not a.no_write:
        cfg_path = WEIGHTS_DIR / 'config.json'
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        cfg['conf'] = best[1]
        cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
        print(f'[tune] wrote {cfg_path}')


if __name__ == '__main__':
    main()
