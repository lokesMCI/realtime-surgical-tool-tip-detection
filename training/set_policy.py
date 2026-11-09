"""Switch the output policy in weights/config.json without retraining (the score weighs precision 0.40, recall 0.35).

    uv run python training/set_policy.py balanced            # one threshold for all classes (what scored 0.629)
    uv run python training/set_policy.py bipolar_only        # only the most reliable class, high confidence
    uv run python training/set_policy.py high_conf           # all classes, precision-first thresholds
    uv run python training/set_policy.py custom --bipolar 0.5 --scissors 0.8 --suction 0.7 [--top-k 1]

A class with a threshold above 1 is never output. The organisers confirmed precision-first policies are allowed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CFG = Path(__file__).resolve().parents[1] / 'weights' / 'config.json'
OFF = 1.01
PRESETS = {
    'balanced': {'conf': 0.3, 'conf_by_class': None, 'top_k': None},
    'bipolar_only': {'conf': 0.5, 'conf_by_class': {'bipolar': 0.5, 'scissors': OFF, 'suction': OFF}, 'top_k': None},
    'high_conf': {'conf': 0.6, 'conf_by_class': {'bipolar': 0.6, 'scissors': 0.6, 'suction': 0.7}, 'top_k': None},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy', choices=[*PRESETS, 'custom'])
    for c in ('bipolar', 'scissors', 'suction'):
        ap.add_argument(f'--{c}', type=float, default=OFF, help='custom: threshold (default off)')
    ap.add_argument('--top-k', type=int, default=None)
    a = ap.parse_args()
    pol = PRESETS.get(a.policy) or {'conf': min(a.bipolar, a.scissors, a.suction),
                                    'conf_by_class': {'bipolar': a.bipolar, 'scissors': a.scissors, 'suction': a.suction},
                                    'top_k': a.top_k}
    cfg = json.loads(CFG.read_text())
    cfg.update(pol)
    CFG.write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'[policy] {a.policy}: {json.dumps(cfg)}')


if __name__ == '__main__':
    main()
