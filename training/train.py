"""Train a YOLO-pose tip detector on the dataset made by prep_data.py, optionally ship the result to weights/.

    uv run python training/train.py --data work/yolo/data.yaml --model yolo26s-pose.pt --epochs 60 --imgsz 640 --name base_s

One change per run, name every run, and compare them on the by-clip val split (training/tune_threshold.py prints the
leaderboard-style score), never on the live leaderboard.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def pick_device():
    import torch
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def strong_photometric():
    """Camera/scene randomisation: the hidden test is other videos (other light, focus, compression), not other frames."""
    import albumentations as A
    return [
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.RandomGamma(p=0.3),
        A.CLAHE(p=0.1),
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7)), A.Defocus(radius=(2, 4))], p=0.3),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(40, 95), p=0.3),
        A.ToGray(p=0.05),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='work/yolo/data.yaml')
    ap.add_argument('--model', default='yolo26s-pose.pt', help='COCO-pose pretrained start (downloaded on first use), or a .pt of yours')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--name', default='base')
    ap.add_argument('--project', default='runs/surgnet')
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--lr0', type=float, default=0.001, help='AdamW; small data and a pretrained start want a low rate')
    ap.add_argument('--pose', type=float, default=None, help='keypoint loss gain (Ultralytics default 12)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--aug', choices=['base', 'strong'], default='base',
                    help='strong: blur/noise/JPEG/contrast/gamma/grey (Albumentations) + wider geometry, for unseen videos')
    ap.add_argument('--ship', action='store_true', help='copy best.pt to weights/best.pt and write weights/config.json')
    a = ap.parse_args()

    from ultralytics import YOLO
    device = a.device if a.device is not None else pick_device()
    model_path = Path(a.model)
    if not model_path.exists() and model_path.parent == Path('.'):       # bare name: keep downloads out of the repo root
        model_path = Path('work/pretrained') / model_path.name
        model_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    extra = {'pose': a.pose} if a.pose is not None else {}
    geometry = dict(degrees=20, translate=0.1, scale=0.5)
    if a.aug == 'strong':
        extra['augmentations'] = strong_photometric()
        geometry = dict(degrees=30, translate=0.15, scale=0.6, mixup=0.1)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device, workers=a.workers,
        project=str(Path(a.project).resolve()), name=a.name, exist_ok=True, patience=a.patience, optimizer='AdamW', lr0=a.lr0, seed=a.seed,
        # operating-theatre video has no fixed 'up': flip both ways; colour/brightness jitter for the sim-to-real gap
        fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=10, **geometry,
        hsv_h=0.03, hsv_s=0.6, hsv_v=0.5, erasing=0.2, plots=True, **extra)

    best = Path(model.trainer.best)
    print(f'[train] best weights: {best}')
    if a.ship:
        dst = Path('weights')
        dst.mkdir(exist_ok=True)
        shutil.copy2(best, dst / 'best.pt')
        cfg_path = dst / 'config.json'
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        cfg.update({'weights': 'best.pt', 'imgsz': a.imgsz})
        cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
        print(f'[train] shipped -> {dst / "best.pt"}  (now run training/tune_threshold.py)')


if __name__ == '__main__':
    main()
