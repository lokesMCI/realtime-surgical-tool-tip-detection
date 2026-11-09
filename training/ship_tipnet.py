"""Put a trained TipNet into weights/ and point weights/config.json at it, so `SurgNet()` (the graded entry point) loads it.

    uv run python training/ship_tipnet.py --ckpt runs/tipnet/tip_s_all/best.tipnet.pt --conf 0.3

Stores the state dict in fp16 (half the size; SurgNet casts it back on load). The confidence threshold must come from a
held-out-video run (training/eval_ckpt.py prints it), never from the leaderboard.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

WEIGHTS = Path(__file__).resolve().parents[1] / 'weights'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--conf', type=float, required=True)
    ap.add_argument('--name', default='surgnet', help='weights/<name>.tipnet.pt')
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    assert ck.get('arch') == 'tipnet', f'{a.ckpt} is not a TipNet checkpoint'
    ck['state_dict'] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck['state_dict'].items()}
    dst = WEIGHTS / f'{a.name}.tipnet.pt'
    torch.save(ck, dst)
    cfg_path = WEIGHTS / 'config.json'
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    cfg.update({'weights': dst.name, 'conf': a.conf, 'conf_by_class': None, 'max_det': 10})
    cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'[ship] {dst} ({dst.stat().st_size / 1e6:.1f} MB); config.json -> {cfg}')


if __name__ == '__main__':
    main()
