"""Train TipNet (DINOv2 ViT-S/14 + tip heatmaps) on a prep_data.py / make_composites.py dataset folder.

    uv run python training/train_tipnet.py --data work/yolo_v --name tip_s --epochs 30
    uv run python training/eval_ckpt.py --weights runs/tipnet/tip_s/best.tipnet.pt --images work/yolo_v/val_images --ann ...

Reads the YOLO-pose label files (class + tip keypoint) so the by-video split and the composites are reused as they are.
Each epoch ends with a held-out score using the organisers' matcher (leaderboard formula without fps, conf sweep), and
the best epoch is kept as best.tipnet.pt (ship it as weights/<name>.tipnet.pt and point config.json at it).
"""
from __future__ import annotations
import argparse
import math
import random
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from surgnet.inference.tipnet import MEAN, STD, TipNet, decode
from surgnet.scoring import compute_metrics, match_points
from surgnet.shared.detection import CLASSES

PAD_BGR = tuple(int(255 * m) for m in MEAN[::-1])          # padding = ImageNet mean, i.e. 0 after normalisation


def read_labels(label_path: Path, w: int, h: int):
    """[(cls, x, y)] tips in pixels from a YOLO-pose label file."""
    out = []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            v = line.split()
            if len(v) >= 7:
                out.append((int(v[0]), float(v[5]) * w, float(v[6]) * h))
    return out


class TipData(Dataset):
    def __init__(self, root: Path, split: str, size: int, stride: float, train: bool, sigma: float):
        self.imgs = sorted(p for p in (root / 'images' / split).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        self.lab = root / 'labels' / split
        self.size, self.stride, self.train, self.sigma = size, stride, train, sigma
        self.out = int(round(size / stride))

    def __len__(self):
        return len(self.imgs)

    def affine(self, w, h):
        """2x3 matrix: frame -> network input. Eval: plain letterbox. Train: + scale/rotate/flip/translate."""
        r = self.size / max(w, h)
        if not self.train:
            return np.array([[r, 0, 0], [0, r, 0]], np.float32)
        s = r * random.uniform(0.6, 1.5)
        a = math.radians(random.uniform(-30, 30))
        fx = -1 if random.random() < 0.5 else 1
        fy = -1 if random.random() < 0.5 else 1
        R = np.array([[math.cos(a) * fx, -math.sin(a) * fy], [math.sin(a) * fx, math.cos(a) * fy]], np.float32) * s
        c_src = np.array([w / 2, h / 2], np.float32)
        c_dst = np.array([self.size / 2, self.size / 2], np.float32) + np.random.uniform(-0.2, 0.2, 2).astype(np.float32) * self.size
        t = c_dst - R @ c_src
        return np.hstack([R, t[:, None]]).astype(np.float32)

    def photometric(self, img):
        img = img.astype(np.float32)
        img = img * random.uniform(0.7, 1.3) + random.uniform(-25, 25)            # contrast / brightness
        img = img * np.random.uniform(0.9, 1.1, 3).astype(np.float32)             # white balance
        img = np.clip(img, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int) + random.randint(-6, 6)) % 180
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if random.random() < 0.25:
            k = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.2:
            q = random.randint(40, 90)
            img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        return img

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        tips = read_labels(self.lab / f'{p.stem}.txt', w, h)
        M = self.affine(w, h)
        x = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_LINEAR, borderValue=PAD_BGR)
        if self.train:
            x = self.photometric(x)
        x = (x[..., ::-1].astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()

        heat = np.zeros((len(CLASSES), self.out, self.out), np.float32)
        off = np.zeros((2, self.out, self.out), np.float32)
        mask = np.zeros((self.out, self.out), np.float32)
        yy, xx = np.mgrid[0:self.out, 0:self.out]
        for c, tx, ty in tips:
            gx, gy = (M @ np.array([tx, ty, 1.0], np.float32)) / self.stride
            if not (0 <= gx < self.out and 0 <= gy < self.out):
                continue
            ix, iy = int(gx), int(gy)
            g = np.exp(-((xx - gx + 0.5) ** 2 + (yy - gy + 0.5) ** 2) / (2 * self.sigma ** 2))
            heat[c] = np.maximum(heat[c], g)
            heat[c, iy, ix] = 1.0
            off[:, iy, ix] = (gx - ix, gy - iy)
            mask[iy, ix] = 1.0
        return x, torch.from_numpy(heat), torch.from_numpy(off), torch.from_numpy(mask)


def focal_loss(logits, gt):
    """CenterNet penalty-reduced focal loss."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


@torch.no_grad()
def evaluate(model, ds, device, stride):
    """Leaderboard objective (no fps) on the held-out folder, best over a conf sweep; tips compared in input pixels."""
    model.eval()
    preds, gts = [], []
    for i, p in enumerate(ds.imgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        x, *_ = ds[i]
        r = ds.size / max(w, h)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            heat, off = model(x[None].to(device))
        preds.append([(c, gx / r, gy / r, s) for c, gx, gy, s in decode(heat, off.float(), stride, conf=0.05)[0]])
        gts.append((read_labels(ds.lab / f'{p.stem}.txt', w, h), math.hypot(w, h)))
    best = (-1, 0, None)
    for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
        per_image = []
        for pr, (gt, diag) in zip(preds, gts):
            res = {}
            for c, name in enumerate(CLASSES):
                res[name] = match_points([(x, y) for cc, x, y, s in pr if cc == c and s >= thr],
                                         [(x, y) for cc, x, y in gt if cc == c], 0.025 * diag)
            per_image.append(res)
        o = compute_metrics(per_image)['overall']
        err = o['median_error_px']
        obj = 0.4 * o['precision'] + 0.35 * o['recall'] + 0.15 * (0 if err is None else max(0, 1 - err / 50))
        if obj > best[0]:
            best = (obj, thr, o)
    model.train()
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='work/yolo_v')
    ap.add_argument('--name', default='tip_s')
    ap.add_argument('--backbone', default='vit_small_patch14_dinov2.lvd142m')
    ap.add_argument('--size', type=int, default=518, help='network input side, multiple of 14')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3, help='head; the backbone gets lr * --backbone-lr-mult')
    ap.add_argument('--backbone-lr-mult', type=float, default=0.05)
    ap.add_argument('--sigma', type=float, default=2.0, help='heatmap Gaussian, in output cells (3.5 px each)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no-val', action='store_true', help='final run on all data: save the last epoch')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path('runs/tipnet') / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = (5, 8, 11)
    model = TipNet(a.backbone, n_cls=len(CLASSES), size=a.size, pretrained=True, layers=layers).to(device)
    stride = model.stride
    root = Path(a.data)
    train_ds = TipData(root, 'train', a.size, stride, True, a.sigma)
    val_ds = TipData(root, 'val', a.size, stride, False, a.sigma)
    dl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                    persistent_workers=a.workers > 0)
    bb = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': bb, 'lr': a.lr * a.backbone_lr_mult}, {'params': head, 'lr': a.lr}],
                            weight_decay=1e-4)
    total = a.epochs * len(dl)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm))))

    def save(path):
        torch.save({'arch': 'tipnet', 'backbone': a.backbone, 'size': a.size, 'layers': list(layers),
                    'names': list(CLASSES), 'state_dict': model.state_dict()}, path)

    print(f'[tipnet] train {len(train_ds)}  val {len(val_ds)}  device {device}  out {out_dir}', flush=True)
    best = -1
    for ep in range(a.epochs):
        t0, tot = time.time(), 0.0
        for x, heat, off, mask in dl:
            x, heat, off, mask = (t.to(device, non_blocking=True) for t in (x, heat, off, mask))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                ph, po = model(x)
            l_heat = focal_loss(ph, heat)
            l_off = (F.l1_loss(po.float(), off, reduction='none').sum(1) * mask).sum() / mask.sum().clamp(min=1)
            loss = l_heat + l_off
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f'[tipnet] epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.3f} {time.time() - t0:.0f}s'
        if not a.no_val and len(val_ds):
            obj, thr, o = evaluate(model, val_ds, device, stride)
            msg += f'  val obj {obj:.3f} @conf {thr}  P {o["precision"]:.2f} R {o["recall"]:.2f} err {o["median_error_px"] or 0:.1f}'
            if obj > best:
                best = obj
                save(out_dir / 'best.tipnet.pt')
                msg += '  *'
        print(msg, flush=True)
        save(out_dir / 'last.tipnet.pt')
    if a.no_val:
        save(out_dir / 'best.tipnet.pt')
    print(f'[tipnet] done; best val objective {best:.3f}; weights {out_dir}/best.tipnet.pt', flush=True)


if __name__ == '__main__':
    sys.exit(main())
