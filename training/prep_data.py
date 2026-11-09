"""CVAT XML + frames -> Ultralytics YOLO-pose dataset (one keypoint per tool = the tip).

    uv run python training/prep_data.py --images data/images --ann data/annotations.xml --out work/yolo

Why pose: the task scores a tip POINT per tool, so each tool becomes one object (class + box) carrying one keypoint.
The ground-truth tip is built exactly like the scorer builds it (surgnet.labelio.cvat): points of one tool share a
CVAT `group_id`, occluded points are dropped, the rest are averaged. The box comes from the tool's mask bbox.

Split is by VIDEO, never by frame or timestamp: frames of one video are near-duplicates, so anything finer leaks and makes
validation look far better than the hidden test (by-timestamp val said 0.79, the leaderboard said 0.42).

Writes: <out>/{images,labels}/{train,val}/, <out>/data.yaml, <out>/val_images/<name> (a name-preserving symlink tree
you can hand to benchmark.py --data together with the full annotations.xml, which then scores only the val frames).
"""
from __future__ import annotations
import argparse
import collections
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ('suction', 'bipolar', 'scissors')        # must match surgnet.shared.detection.CLASSES
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_EXT = ('.jpg', '.jpeg', '.png')


def clip_key(name: str) -> str:
    """The unit that must never straddle train and val: one VIDEO (capture), not one timestamp.

    Frame names carry a time and a video: in '<time>_L_<video>.jpg' the number BEFORE _L is the time and the one after
    it is the video (frames of one video are near-identical; frames sharing only the time are not). Splitting by time
    leaks every held-out video into training. The other recognised patterns:
        '<domain>/<time>_L_<k>.jpg'     -> '<domain>/L<k>'
        '<domain>/frame_<n>.png'        -> '<domain>/f<n // 4000>'   (one long video cut into blocks)
        '<domain>/<video_id>_<n>.jpg'   -> '<domain>/<video_id>'
    """
    folder, base = name.split('/', 1) if '/' in name else ('', name)
    stem = Path(base).stem
    m = re.match(r'\d+_L(?:_(\d+))?$', stem)
    if m:
        return f'{folder}/L{int(m.group(1) or 0)}'
    m = re.match(r'frame_(\d+)', stem)
    if m:
        return f'{folder}/f{int(m.group(1)) // 4000}'
    m = re.match(r'(.+?)_\d+$', stem)
    if m:
        return f'{folder}/{m.group(1)}'
    return f'{folder}/{stem}'


def parse_instances(xml_path) -> dict:
    """{image name: {'w', 'h', 'tools': [{'cls', 'tip', 'box'}], 'skipped': n}} for every <image> (empty frames too)."""
    root = ET.parse(str(xml_path)).getroot()
    out = {}
    for im in root.iter('image'):
        name, w, h = im.get('name'), float(im.get('width')), float(im.get('height'))
        groups = collections.OrderedDict()                       # (label, group key) -> {'pts': [...], 'box': ...}
        for el in im:
            label = el.get('label')
            if label not in CLASS_ID:
                continue
            key = (label, el.get('group_id') if el.get('group_id') is not None else ('el', id(el)))
            g = groups.setdefault(key, {'pts': [], 'box': None, 'seen_points': False})
            if el.tag == 'mask':
                x0, y0 = float(el.get('left')), float(el.get('top'))
                g['box'] = (x0, y0, x0 + float(el.get('width')), y0 + float(el.get('height')))
            elif el.tag == 'points':
                g['seen_points'] = True
                if el.get('occluded', '0') == '1':
                    continue
                for pair in el.get('points', '').split(';'):
                    if pair.strip():
                        x, y = pair.split(',')
                        g['pts'].append((float(x), float(y)))
        tools, skipped = [], 0
        for (label, _), g in groups.items():
            if not g['pts']:                                      # mask only, or tip occluded: scorer ignores it too
                skipped += 1
                continue
            tip = (sum(p[0] for p in g['pts']) / len(g['pts']), sum(p[1] for p in g['pts']) / len(g['pts']))
            tools.append({'cls': CLASS_ID[label], 'tip': tip, 'box': g['box']})
        out[name] = {'w': w, 'h': h, 'tools': tools, 'skipped': skipped}
    return out


def make_box(tool, w, h, mode, tip_box_frac):
    """Pixel box (x0, y0, x1, y1). 'mask': tool mask bbox grown to contain the tip. 'tip': square around the tip."""
    tx, ty = tool['tip']
    if mode == 'mask' and tool['box'] is not None:
        x0, y0, x1, y1 = tool['box']
        pad = 4.0
        x0, y0, x1, y1 = min(x0, tx - pad), min(y0, ty - pad), max(x1, tx + pad), max(y1, ty + pad)
    else:
        half = tip_box_frac * min(w, h) / 2
        x0, y0, x1, y1 = tx - half, ty - half, tx + half, ty + half
    return max(0.0, x0), max(0.0, y0), min(w, x1), min(h, y1)


def label_line(tool, w, h, mode, tip_box_frac) -> str:
    x0, y0, x1, y1 = make_box(tool, w, h, mode, tip_box_frac)
    tx = min(max(tool['tip'][0], 0.0), w)
    ty = min(max(tool['tip'][1], 0.0), h)
    return (f"{tool['cls']} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f} "
            f"{tx / w:.6f} {ty / h:.6f} 2")


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def assign_split(clips, val_every, val_clips):
    """Deterministic and per folder: about 1/val_every of the clips (at least one) go to val, spread evenly through the
    sorted order. a folder with only two videos holds one whole video out: the honest sim-to-real check."""
    by_folder = collections.defaultdict(list)
    for c in sorted(set(clips)):
        by_folder[c.split('/')[0]].append(c)
    val = set(val_clips)
    if val_every:
        for cs in by_folder.values():
            n_val = max(1, round(len(cs) / val_every))
            step = len(cs) / n_val
            val.update(cs[int(step * (i + 0.5))] for i in range(n_val))
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='data/images', help='root holding <folder>/<frame> exactly as names in the XML')
    ap.add_argument('--ann', default='data/annotations.xml')
    ap.add_argument('--out', default='work/yolo')
    ap.add_argument('--box', choices=['mask', 'tip'], default='mask', help='box source (mask bbox, or a square at the tip)')
    ap.add_argument('--tip-box-frac', type=float, default=0.18, help='--box tip: side as a fraction of min(W,H)')
    ap.add_argument('--val-every', type=int, default=5, help='1 in N clips per folder goes to val, at least one per folder (0 = none)')
    ap.add_argument('--val-clips', nargs='*', default=[], help='extra clip keys forced into val, e.g. domain/L3')
    ap.add_argument('--full', action='store_true', help='final run: train on everything (val keeps the same frames, so '
                                                          'its numbers are then optimistic; use it only after the val run)')
    ap.add_argument('--folders', nargs='*', default=None, help='only these image folders (default: all that exist)')
    a = ap.parse_args()

    images_root, out = Path(a.images), Path(a.out)
    inst = parse_instances(a.ann)
    present = {}                                                  # name -> path on disk
    for name in inst:
        p = images_root / name
        if p.is_file():
            present[name] = p
    missing = [n for n in inst if n not in present]
    if a.folders:
        present = {n: p for n, p in present.items() if n.split('/')[0] in a.folders}
    if not present:
        raise SystemExit(f'no frame from {a.ann} found under {images_root} (looked for e.g. {next(iter(inst))})')

    val_clips = assign_split([clip_key(n) for n in present], a.val_every, a.val_clips)
    if out.exists():
        shutil.rmtree(out)
    stats = collections.Counter()
    for name, path in sorted(present.items()):
        rec = inst[name]
        split = 'val' if clip_key(name) in val_clips else 'train'
        flat = name.replace('/', '__')
        stem = Path(flat).stem
        ext = Path(flat).suffix.lower()
        lines = [label_line(t, rec['w'], rec['h'], a.box, a.tip_box_frac) for t in rec['tools']]
        splits = ['train', 'val'] if (a.full and split == 'val') else [split]
        for s in splits:
            link(path, out / 'images' / s / f'{stem}{ext}')
            lab = out / 'labels' / s / f'{stem}.txt'
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text('\n'.join(lines) + ('\n' if lines else ''))
        if split == 'val':
            link(path, out / 'val_images' / name)
        stats[split, name.split('/')[0], 'frames'] += 1
        stats[split, name.split('/')[0], 'tools'] += len(rec['tools'])
        stats['skipped_instances'] += rec['skipped']
        stats[split, 'empty'] += 0 if lines else 1

    (out / 'data.yaml').write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"kpt_shape: [1, 3]\nflip_idx: [0]\nnames:\n" + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))

    print(f'[prep] {len(present)} frames -> {out}  (missing on disk: {len(missing)}; val groups: {sorted(val_clips)})')
    for k in sorted(k for k in stats if isinstance(k, tuple) and len(k) == 3):
        print(f'  {k[0]:5s} {k[1]:15s} {k[2]:6s} {stats[k]}')
    print(f'  empty frames  train={stats["train", "empty"]} val={stats["val", "empty"]}  '
          f'instances without a visible tip (skipped)={stats["skipped_instances"]}')
    print(f'  data.yaml: {out / "data.yaml"}   val frames for benchmark.py --data: {out / "val_images"}')


if __name__ == '__main__':
    main()
