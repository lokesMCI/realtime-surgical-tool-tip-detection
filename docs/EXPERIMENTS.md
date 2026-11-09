# Experiment log

Scores come from the challenge's matcher and leaderboard formula. "Held-out" means whole videos that the model never saw;
section 2's objective column leaves out the fps term; section 3 includes it (0.10 for every model here, all are above
100 fps). Only aggregate numbers are reported.

## 1. The validation split was leaking

The first models were validated on a split by frame-name prefix. That prefix is a time stamp, the video id is the suffix,
so every held-out frame had near-twins from the same video in training. Check: the next frame of the same video differs
much less (mean absolute pixel difference about 7-11 on a 64x72 thumbnail) than a frame from another video at the same
time (about 22-24). Consequence: the
validation score said 0.79, the leaderboard said 0.42. Everything below uses a split by video.

## 2. Models on held-out videos

| Model | Held-out objective | Notes |
|---|---|---|
| YOLO26s-pose (tip as keypoint), default augmentation | 0.65 | best epoch is early; overfits afterwards |
| YOLO26s-pose, strong augmentation (blur, noise, JPEG, gamma, wider geometry, mixup) | 0.56 | worse: the extra variation cost accuracy on the training domains |
| TipNet, DINOv2 ViT-S/14 | **0.75** | better on every domain and class, half the latency |
| TipNet, DINOv2 ViT-B/14 | 0.77-0.78 | better still on held-out videos, same latency |
| TipNet ViT-S at 728 px instead of 518 px | no better | worse tip error on real footage |

Cross-class de-duplication (keep only the surer class when two classes fire on one tip): no gain.

## 3. Leaderboard (test footage) vs held-out

| Submission | Held-out score (incl. fps term) | Test | Comment |
|---|---|---|---|
| TipNet ViT-S, all classes at one threshold | 0.85 (benchmark.py) | 0.629 | test precision 0.55: half the predictions wrong |
| TipNet ViT-B, top-1 tool per frame | 0.87 | 0.581 | recall fell while precision stayed 0.56 |
| TipNet ViT-S, bipolar only | 0.67 | 0.670 | precision 0.986, recall 0.18, bipolar F1 0.75 |

Reading: on the test, the model's scissors and suction predictions were mostly wrong even at high confidence, so tighter
thresholds or top-1 did not help, while dropping those classes did. Held-out videos are not the test distribution:
the gap (0.85 vs 0.63) shows how different the unseen footage is, and held-out numbers should rank models, not predict
leaderboard scores.

## 4. The output policy is a free lever

With precision weighted 0.40 and recall 0.35, a class the model finds with ~99 % precision but ~20 % recall can beat
a balanced output. Measured on held-out videos before trying it on the test: per-class precision vs confidence curves,
then policy options (single threshold, per-class thresholds, top-k, classes switched off). The policy lives in
`weights/config.json` and `training/set_policy.py` changes it without retraining.

## 5. Test-time mirroring

Averaging the heatmap of the frame and its mirror images (one batched pass) tightened the tip location and removed a
few false positives. Bipolar-only at confidence 0.5 on held-out videos: precision 0.978 to 1.000 with horizontal flip;
median tip error 6.0 px without mirroring, 5.6 px with four-way mirroring (real footage 17.3 to 13.7 px); about +1 to +2 ms
on an idle GPU.

## 6. Not finished

- Scissors and suction on unseen footage (the test F1 for scissors stayed around 0.4).
- Copy-paste composites (`training/make_composites.py`) were generated and used with the YOLO baseline only, not
  evaluated with TipNet.
- No external surgical datasets were used.
