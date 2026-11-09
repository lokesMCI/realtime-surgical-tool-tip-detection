#!/usr/bin/env bash
# One command on a GPU box: training data -> TipNet -> weights/ -> benchmark -> contract test.
#
#   cd <repo root>
#   tmux new -s train 'bash scripts/brev_train_submit.sh 2>&1 | tee work/brev_train_submit.log'
#
# Stages (each prints a ✅ line):
#   1. env + data link        uv sync; data/images -> ~/data/train/images; the CVAT annotations.xml
#   2. honest run             split by VIDEO, train TipNet, score held-out videos -> picks the confidence threshold
#   3. final run              same recipe on ALL videos (more data, no val), fixed epoch count
#   4. ship                   weights/<name>.tipnet.pt + weights/config.json (SurgNet() loads exactly this)
#   5. check                  organiser benchmark on the held-out videos + pytest contract test
# Env overrides: EPOCHS=30 BACKBONE=vit_small_patch14_dinov2.lvd142m SKIP_HONEST=1 CONF=0.3 (needed with SKIP_HONEST)
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/.."
DATA=${DATA:-$HOME/data/train}
ANN=$DATA/annotations.xml
EPOCHS=${EPOCHS:-30}
BACKBONE=${BACKBONE:-vit_small_patch14_dinov2.lvd142m}
TAG=${TAG:-$(date +%H%M)}
mkdir -p work

echo "== 1. environment and data"
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('✅ GPU', torch.cuda.get_device_name(0))"
ln -sfn "$DATA/images" data/images
ls data/images

uv run python training/prep_data.py --images data/images --ann "$ANN" --out work/yolo_v           # held-out videos
uv run python training/prep_data.py --images data/images --ann "$ANN" --out work/yolo_all --full  # everything

if [ "${SKIP_HONEST:-0}" != "1" ]; then
  echo "== 2. honest run (held-out videos)"
  uv run python training/train_tipnet.py --data work/yolo_v --name honest_$TAG --epochs "$EPOCHS" --backbone "$BACKBONE"
  uv run python training/eval_ckpt.py --weights runs/tipnet/honest_$TAG/best.tipnet.pt --images work/yolo_v/val_images \
      --ann "$ANN" --out work/res_honest_$TAG.json
  CONF=$(python3 -c "import json; print(json.load(open('work/res_honest_$TAG.json'))['conf'])")
fi
CONF=${CONF:?set CONF when SKIP_HONEST=1}
echo "✅ threshold from held-out videos: $CONF"

echo "== 3. final run on all videos"
uv run python training/train_tipnet.py --data work/yolo_all --name final_$TAG --epochs "$EPOCHS" --backbone "$BACKBONE" --no-val

echo "== 4. ship"
uv run python training/ship_tipnet.py --ckpt runs/tipnet/final_$TAG/best.tipnet.pt --conf "$CONF"

echo "== 5. check what the organisers will run"
uv run pytest -q tests/test_contract.py
# held-out videos were in the final training set, so this score is optimistic: it checks loading, speed and format
uv run python benchmark.py --data work/yolo_v/val_images --labels "$ANN" --out work/results_final_$TAG.json
echo "✅ done. Commit: git add weights/config.json weights/*.tipnet.pt && git commit -m 'TipNet $TAG' && git push"
