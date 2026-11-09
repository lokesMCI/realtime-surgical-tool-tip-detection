"""Copy-paste composites: cut tools out with their CVAT masks and paste them onto frames of OTHER videos/domains.

    uv run python training/make_composites.py --base work/yolo_v --ann ~/data/train/annotations.xml \
        --images data/images --out work/yolo_vc --n 1500

Why: the hidden test is unseen videos. The model must recognise a tool by its shape, not by the scene it was filmed in,
and scissors/bipolar are rare in real footage. Only TRAIN-split frames are used as tool sources and backgrounds (the val
split of --base is copied unchanged), so validation stays honest.

Each composite: one train frame as background (its own labels kept unless a pasted tool covers that tip), plus 1-2 pasted
tools (rare classes oversampled), each randomly flipped, rotated about its tip, scaled, colour-jittered and feathered.
The pasted tool's keypoint is its transformed tip; its box is the transformed mask's bounding box.
"""
from __future__ import annotations
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

from prep_data import CLASS_ID, CLASSES, clip_key, label_line

RARE_BOOST = {'suction': 1.0, 'bipolar': 2.0, 'scissors': 3.0}   # sampling weight per pasted class


def decode_cvat_rle(rle: str, w: int, h: int) -> np.ndarray:
    """CVAT 1.1 mask: run lengths over the box (w x h), row-major, starting with a background run."""
    counts = [int(c) for c in rle.split(',') if c.strip()]
    flat = np.zeros(w * h, np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape(h, w)


def load_tools(ann, train_names):
    """[{name, cls, tip, mask (full-frame-offset crop), x0, y0}] for tools with a visible tip and a mask, train frames only."""
    root = ET.parse(str(ann)).getroot()
    tools = []
    for im in root.iter('image'):
        name = im.get('name')
        if name not in train_names:
            continue
        groups = defaultdict(lambda: {'pts': [], 'masks': []})
        for el in im:
            label = el.get('label')
            if label not in CLASS_ID:
                continue
            g = groups[(label, el.get('group_id') or ('el', id(el)))]
            if el.tag == 'mask':
                x0, y0, w, h = (int(float(el.get(k))) for k in ('left', 'top', 'width', 'height'))
                g['masks'].append((x0, y0, decode_cvat_rle(el.get('rle'), w, h)))
            elif el.tag == 'points' and el.get('occluded', '0') != '1':
                for pair in el.get('points', '').split(';'):
                    if pair.strip():
                        x, y = pair.split(',')
                        g['pts'].append((float(x), float(y)))
        for (label, _), g in groups.items():
            if not g['pts'] or not g['masks']:
                continue
            x0 = min(m[0] for m in g['masks']); y0 = min(m[1] for m in g['masks'])
            x1 = max(m[0] + m[2].shape[1] for m in g['masks']); y1 = max(m[1] + m[2].shape[0] for m in g['masks'])
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            for mx, my, m in g['masks']:
                mask[my - y0:my - y0 + m.shape[0], mx - x0:mx - x0 + m.shape[1]] |= m
            tip = tuple(np.mean(g['pts'], axis=0))
            tools.append({'name': name, 'cls': label, 'tip': tip, 'mask': mask, 'x0': x0, 'y0': y0})
    return tools


def tip_mask_distance(t) -> float:
    """Pixels from the tip to the nearest mask pixel (0 = on the mask): sanity check of the RLE decoding."""
    ys, xs = np.nonzero(t['mask'])
    if len(xs) == 0:
        return float('inf')
    return float(np.min(np.hypot(xs + t['x0'] - t['tip'][0], ys + t['y0'] - t['tip'][1])))


def paste(bg, tool, src_img, rng):
    """Paste `tool` (cut from src_img) into bg in place. Returns (cls, tip, box, alpha_full) or None if it fell outside."""
    H, W = bg.shape[:2]
    m = tool['mask']
    x0, y0 = tool['x0'], tool['y0']
    crop = src_img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
    if crop.shape[:2] != m.shape:                                  # mask box clipped by the frame edge
        m = m[:crop.shape[0], :crop.shape[1]]
    alpha = cv2.erode(m * 255, np.ones((3, 3), np.uint8))          # drop the green / tissue fringe
    tip = np.array([tool['tip'][0] - x0, tool['tip'][1] - y0])

    # colour jitter of the tool only (lighting differs between videos)
    crop = crop.astype(np.float32)
    crop = crop * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    crop = np.clip(crop, 0, 255).astype(np.uint8)

    scale = rng.uniform(0.6, 1.4) * (min(H, W) / min(src_img.shape[:2]))
    angle = rng.uniform(-45, 45)
    if rng.random() < 0.5:                                         # horizontal flip about the crop
        crop, alpha = crop[:, ::-1].copy(), alpha[:, ::-1].copy()
        tip[0] = crop.shape[1] - 1 - tip[0]
    # destination tip: anywhere in the inner 80 % of the frame
    dst_tip = np.array([rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H])
    M = cv2.getRotationMatrix2D((float(tip[0]), float(tip[1])), angle, scale)
    M[:, 2] += dst_tip - tip
    warped = cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    a = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    a = cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    if a.sum() < 200:
        return None
    bg[:] = (warped * a + bg * (1 - a)).astype(np.uint8)
    ys, xs = np.nonzero(a[..., 0] > 0.5)
    if len(xs) == 0:
        return None
    box = (min(xs.min(), dst_tip[0]), min(ys.min(), dst_tip[1]), max(xs.max(), dst_tip[0]), max(ys.max(), dst_tip[1]))
    return tool['cls'], (float(dst_tip[0]), float(dst_tip[1])), box, a[..., 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='work/yolo_v', help='dataset from prep_data.py (its val split is kept as is)')
    ap.add_argument('--ann', default='data/annotations.xml')
    ap.add_argument('--images', default='data/images')
    ap.add_argument('--out', default='work/yolo_vc')
    ap.add_argument('--n', type=int, default=1500, help='composites to generate')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--check', action='store_true', help='only check mask decoding and print source stats')
    a = ap.parse_args()

    base, out, images = Path(a.base), Path(a.out), Path(a.images)
    train_stems = {p.stem for p in (base / 'images' / 'train').iterdir()}
    from prep_data import parse_instances
    inst = parse_instances(a.ann)
    train_names = {n for n in inst if Path(n.replace('/', '__')).stem in train_stems}
    tools = load_tools(a.ann, train_names)
    d = np.array([tip_mask_distance(t) for t in tools])
    print(f'[composite] {len(tools)} source tools from {len(train_names)} train frames; tip-to-mask distance px: '
          f'median {np.median(d):.1f}, 90% {np.percentile(d, 90):.1f}, >50px: {(d > 50).sum()}')
    for c in CLASSES:
        print(f'   {c:9s} {sum(t["cls"] == c for t in tools)}')
    if a.check:
        return
    tools = [t for t, dist in zip(tools, d) if dist <= 60]            # bipolar/scissors tips sit between arms; farther = bad pair

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, symlinks=True)
    (out / 'data.yaml').write_text((base / 'data.yaml').read_text().replace(str(base.resolve()), str(out.resolve())))
    rng = random.Random(a.seed)
    weights = [RARE_BOOST[t['cls']] for t in tools]
    bg_names = sorted(train_names)
    img_cache = {}

    def read(n):
        if n not in img_cache:
            if len(img_cache) > 300:
                img_cache.clear()
            img_cache[n] = cv2.imread(str(images / n), cv2.IMREAD_COLOR)
        return img_cache[n]

    made = defaultdict(int)
    for i in range(a.n):
        bg_name = rng.choice(bg_names)
        bg = read(bg_name).copy()
        H, W = bg.shape[:2]
        rec = inst[bg_name]
        old = [dict(t) for t in rec['tools']]
        new = []
        for _ in range(rng.choice([1, 1, 2])):
            for _attempt in range(5):
                t = rng.choices(tools, weights=weights)[0]
                if clip_key(t['name']) != clip_key(bg_name):            # paste across videos only
                    break
            res = paste(bg, t, read(t['name']), rng)
            if res is None:
                continue
            cls, tip, box, alpha = res
            # earlier tips now hidden under the pasted tool are no longer visible: drop them
            old = [o for o in old if alpha[min(int(o['tip'][1]), H - 1), min(int(o['tip'][0]), W - 1)] < 0.5]
            new = [o for o in new if alpha[min(int(o['tip'][1]), H - 1), min(int(o['tip'][0]), W - 1)] < 0.5]
            new.append({'cls': CLASS_ID[cls], 'tip': tip, 'box': box})
            made[cls] += 1
        stem = f'comp_{i:05d}'
        cv2.imwrite(str(out / 'images' / 'train' / f'{stem}.jpg'), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [label_line(t, W, H, 'mask', 0.18) for t in old + new]
        (out / 'labels' / 'train' / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    for cache in (out / 'labels').glob('*.cache'):
        cache.unlink()
    print(f'[composite] wrote {a.n} composites to {out}/images/train; pasted per class: {dict(made)}')


if __name__ == '__main__':
    main()
