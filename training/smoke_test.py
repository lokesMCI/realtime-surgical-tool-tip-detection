"""End-to-end plumbing check on GENERATED frames (no real data needed): XML -> prep -> 2 epochs -> SurgNet -> scoring.

    uv run python training/smoke_test.py

It proves the code path works (and prints the leaderboard score of a barely-trained model), not that the model is good.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'work' / 'smoke'
COLORS = {'suction': (200, 200, 200), 'bipolar': (40, 220, 240), 'scissors': (60, 60, 230)}


def make(n_clips=6, frames=4, size=(480, 540)):
    rng = np.random.default_rng(0)
    (W / 'images' / 'synthetic').mkdir(parents=True, exist_ok=True)
    rows = ['<?xml version="1.0" encoding="utf-8"?>', '<annotations>']
    k = 0
    for c in range(n_clips):
        for f in range(frames):
            name = f'synthetic/{c * 30:06d}_L_{f}.jpg'
            img = np.full((size[1], size[0], 3), (40, 140, 40), np.uint8)
            img += rng.integers(0, 25, img.shape, dtype=np.uint8)
            rows.append(f'<image id="{k}" name="{name}" width="{size[0]}" height="{size[1]}">')
            for gid, label in enumerate(rng.permutation(list(COLORS))[:rng.integers(0, 3)], start=1):
                tip = rng.uniform(120, 360), rng.uniform(150, 400)
                base = (tip[0] + rng.uniform(-90, 90), 0)
                cv2.line(img, (int(base[0]), 0), (int(tip[0]), int(tip[1])), COLORS[label], 22)
                x0, y0 = int(min(base[0], tip[0]) - 12), 0
                bw, bh = int(abs(base[0] - tip[0]) + 24), int(tip[1] + 4)
                rows.append(f'<mask label="{label}" source="file" occluded="0" rle="1, 1" left="{max(x0, 0)}" top="{y0}" '
                            f'width="{bw}" height="{bh}" z_order="0" group_id="{gid}"></mask>')
                rows.append(f'<points label="{label}" source="manual" occluded="0" points="{tip[0]:.2f},{tip[1]:.2f}" '
                            f'z_order="0" group_id="{gid}"></points>')
            rows.append('</image>')
            cv2.imwrite(str(W / 'images' / name), img)
            k += 1
    rows.append('</annotations>')
    (W / 'annotations.xml').write_text('\n'.join(rows))


def sh(*cmd):
    print('+', ' '.join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    if W.exists():
        shutil.rmtree(W)
    make()
    py = sys.executable
    sh(py, 'training/prep_data.py', '--images', W / 'images', '--ann', W / 'annotations.xml', '--out', W / 'yolo', '--val-every', '3')
    from ultralytics import YOLO
    (W.parent / 'pretrained').mkdir(parents=True, exist_ok=True)
    m = YOLO(str(W.parent / 'pretrained' / 'yolo26n-pose.pt'))
    m.train(data=str(W / 'yolo' / 'data.yaml'), epochs=2, imgsz=320, batch=8, device='cpu', workers=0,
            project=str(W / 'runs'), name='smoke', exist_ok=True, plots=False, amp=False, verbose=False)
    best = Path(m.trainer.best)
    from surgnet.inference import SurgNet
    from surgnet.benchmark.runner import run
    out = run(W / 'yolo' / 'val_images', labels=W / 'annotations.xml',
              model_factory=lambda **kw: SurgNet(weights=best, **kw), device='cpu')
    print('leaderboard:', out['leaderboard'], '\nlatency:', out['latency'])
    print('SMOKE OK: pipeline runs end to end')


if __name__ == '__main__':
    main()
