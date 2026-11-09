# Real-time surgical tool tip detection

A real-time model that finds the **tip of each surgical instrument** (bipolar forceps, scissors, suction) in a video frame
and names it. Built for the XRlabs Real-Time Vision Challenge (Oxbridge AI X hackathon, 24 October 2026): train on
synthetic surgical video, run frame by frame on a GPU, get scored on unseen footage.

> **Not included, on purpose:** the training data and its labels (covered by an NDA), model weights trained on them, and
> the organisers' interface/benchmark code. This repository is the model code, the training pipeline and the write-up.

## The task and how it is scored

One predicted **point per tool** (class, tip position in pixels, confidence). A prediction counts only if it lies within
2.5 % of the image diagonal of a true tip of the same class (one-to-one Hungarian matching).

    score = 0.40 precision + 0.35 recall + 0.10 min(fps / 100, 1) + 0.15 (1 - median tip error / 50 px)

The model must run offline from shipped weights, and the test footage is never seen in training.

## Result

| Submission | Leaderboard score | What it was |
|---|---|---|
| YOLO26s-pose, tip as a keypoint | 0.42 | first baseline, validated on a leaky split |
| larger YOLO26-pose + scissors oversampling | 0.41 | tuned on the same leaky split |
| TipNet (DINOv2 ViT-S), all classes | 0.629 | honest validation, new architecture |
| TipNet ViT-B, top-1 tool per frame | 0.581 | bigger backbone and tighter output: worse on the test |
| TipNet ViT-S, **bipolar only** (precision-first) | 0.670 | output policy changed, no retraining |
| + horizontal-flip test-time averaging | 0.671 | small gain |

Inference runs at 2.6-4.5 ms per frame (220-390 fps) on a data-centre GPU, so the speed term is saturated.

## What made the difference

1. **A validation split that does not leak.** Frames are named `<time>_L_<video>`, and splitting by the time prefix puts
   near-identical frames of one video on both sides: validation said 0.79 while the leaderboard said 0.42. Holding out
   whole videos made validation rank models correctly and drove every later decision (`training/prep_data.py`, `clip_key`).
2. **Predict the point, not a box.** The baseline was an object detector with one keypoint, so the tip was a regression
   from box anchors. TipNet predicts the tip directly as a heatmap peak with a sub-cell offset.
3. **A backbone that transfers.** DINOv2 (self-supervised, 142M images) beat a COCO-pretrained detector on every
   domain and class, most of all on unseen videos.
4. **Know what the score rewards.** Precision weighs more than recall, and on the test the model's scissors and suction
   predictions were mostly wrong. Outputting only the class it finds reliably raised the score from 0.629 to 0.670 with
   no retraining. The organisers confirmed precision-first output is allowed.
5. **Measure, do not assume.** Heavy augmentation, a larger input size and cross-class de-duplication looked reasonable
   and either did nothing or hurt on held-out videos, and a bigger backbone with a tighter output scored worse on the
   test (see `docs/EXPERIMENTS.md`).

## Architecture (TipNet)

    BGR frame (H, W, 3)
      -> letterbox to 518 x 518 on the GPU (long side scaled, padded), ImageNet normalisation
      -> DINOv2 ViT-S/14 backbone, features from blocks 5, 8, 11            (3 x 384 channels, 37 x 37 grid)
      -> 1x1 conv to 256, then 2 x [2x bilinear upsample + 3x3 conv + BN + ReLU]   (148 x 148 grid, 1 cell = 3.5 px)
      -> heads:  heatmap (3 classes)  +  sub-cell offset (x, y)
      -> peaks (3x3 max-pool) above the class threshold, + offset, back to input pixels, merge same-class duplicates

- **Loss:** CenterNet penalty-reduced focal loss on Gaussian heatmaps + L1 on the offset at the peak.
- **Optimiser:** AdamW, head 1e-3, backbone 5e-5, warm-up then cosine, bf16, 30 epochs.
- **Augmentation:** scale 0.6-1.5, rotation +-30 deg, flips, shifts, brightness/contrast/white balance/hue, blur, JPEG.
- **Test-time:** optional mirror averaging of the heatmaps (one batched pass, offsets mirrored back correctly).
- **Output policy** (per-class thresholds, top-k, mirroring) is configuration in `weights/config.json`.

Code: `src/surgnet/inference/tipnet.py` (model, decoding, mirroring), `training/train_tipnet.py` (training).

## Repository layout

| Path | Purpose |
|---|---|
| `src/surgnet/inference/` | `SurgNet()` entry point and the TipNet model |
| `training/prep_data.py` | CVAT annotations to a YOLO-pose-format dataset, split by video |
| `training/train_tipnet.py` | TipNet training with a held-out score after every epoch |
| `training/eval_ckpt.py`, `per_folder.py`, `tune_threshold.py` | threshold sweeps and per-domain breakdowns with the challenge matcher |
| `training/set_policy.py`, `ship_tipnet.py` | change the output policy / ship a checkpoint without code changes |
| `training/make_composites.py` | copy-paste composites (tools cut out with their masks, pasted on other videos) |
| `training/train.py` | the YOLO26-pose baseline |
| `training/smoke_test.py` | end-to-end plumbing check on generated frames |
| `scripts/brev_train_submit.sh` | one command: split, train, pick the threshold, retrain on all data, ship, check |
| `docs/EXPERIMENTS.md` | the experiment log: what was tried, what it scored, what it taught |

## Running it

The scripts import the challenge's interface package (`surgnet.shared`, `surgnet.labelio`, `surgnet.scoring`: the
`ToolDetection` type, the CVAT reader and the matcher), which is not redistributed here, and need data you are
authorised to use. With those on the path:

    pip install -r requirements.txt
    python training/smoke_test.py                       # generated frames, no real data needed
    bash scripts/brev_train_submit.sh                   # DATA=<dir with images/ and annotations.xml>

## Third-party components

| Component | Licence | Use |
|---|---|---|
| DINOv2 ViT-S/14 and ViT-B/14 weights (via `timm`) | Apache-2.0 | pretrained backbone, fine-tuned |
| Ultralytics YOLO26-pose | AGPL-3.0 | baseline only (`training/train.py`) |
| timm, Albumentations | Apache-2.0, MIT | backbone loading, baseline augmentation |

No external datasets were used. Research and hackathon code, not for clinical use. No licence is granted for this
repository's contents beyond viewing unless one is added.
