"""Break a benchmark results.json down by image folder (domain) and class: where does the score come from?

    uv run python training/per_folder.py work/results_val.json --ann ~/data/train/annotations.xml
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
import numpy as np

from surgnet import ToolDetection
from surgnet.labelio import load_cvat_xml
from surgnet.scoring import match_tips, compute_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results')
    ap.add_argument('--ann', default='data/annotations.xml')
    a = ap.parse_args()
    res, gt = json.load(open(a.results)), load_cvat_xml(a.ann)
    per = defaultdict(list)
    for name, rec in res['images'].items():
        if name not in gt:
            continue
        g = gt[name]
        preds = [ToolDetection(d['class_name'], tuple(d['tip_xy']), d['confidence']) for d in rec['detections']]
        per[name.split('/')[0]].append(match_tips(preds, g.tips, 0.025 * float(np.hypot(g.width, g.height))))
    fps_c = min(1.0, (res['latency']['fps'] or 0) / 100)
    print(f'{"folder":16s} {"frames":>6} {"prec":>6} {"rec":>6} {"err_px":>7} {"score":>6}   per-class recall')
    for folder, items in sorted(per.items()):
        m = compute_metrics(items)
        o = m['overall']
        err = o['median_error_px']
        score = 0.4 * o['precision'] + 0.35 * o['recall'] + 0.1 * fps_c + 0.15 * (0 if err is None else max(0, 1 - err / 50))
        pc = '  '.join(f"{c[:4]}={s['recall']:.2f}(n={s['n_gt']})" for c, s in m['per_class'].items())
        print(f'{folder:16s} {len(items):6d} {o["precision"]:6.2f} {o["recall"]:6.2f} {err or float("nan"):7.1f} {score:6.3f}   {pc}')


if __name__ == '__main__':
    main()
