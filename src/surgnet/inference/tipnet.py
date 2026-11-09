"""TipNet: DINOv2 ViT backbone + CenterNet-style tip heatmaps (one channel per class) + sub-cell offsets.

Why: the leaderboard test is footage unlike any training video. A self-supervised DINOv2 backbone (142M curated images)
carries features that transfer across domains far better than a detector pretrained on COCO people, and a heatmap
predicts the tip itself instead of a box whose corner happens to be the tip.

Geometry: the frame is scaled so its long side is `size` (a multiple of 14), padded right/bottom to size x size, and
the heatmap has stride `size / out` (3.5 px at 518). A tip p (input pixels) lives at grid g = p * r / stride.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TipNet(nn.Module):
    def __init__(self, backbone='vit_small_patch14_dinov2.lvd142m', n_cls=3, size=518, pretrained=False, layers=(5, 8, 11)):
        super().__init__()
        import timm
        self.size, self.n_cls, self.layers = size, n_cls, list(layers)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=size,
                                          dynamic_img_size=True)
        c = self.backbone.embed_dim * len(self.layers)

        def up(cin, cout):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.neck = nn.Sequential(nn.Conv2d(c, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                  up(256, 128), up(128, 64))
        self.heat = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, n_cls, 1))
        self.off = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1))
        nn.init.constant_(self.heat[-1].bias, -2.19)          # CenterNet prior: sigmoid = 0.1

    @property
    def stride(self) -> float:
        return 14 / 4                                          # patch 14, two 2x upsamplings

    def forward(self, x):
        feats = self.backbone.forward_intermediates(x, indices=self.layers, output_fmt='NCHW', intermediates_only=True)
        f = self.neck(torch.cat(feats, 1))
        return self.heat(f), self.off(f)


@torch.no_grad()
def forward_tta(model, x, flips=('h',)):
    """Average heatmaps (in probability space) and offsets over the frame and its mirrored copies, one batched pass.

    A mirrored cell j' maps back to cell W-1-j' and its sub-cell offset o' to 1-o' along the mirrored axis, because the
    whole padded square is mirrored. Returns logits and offsets in the un-mirrored frame, like `model(x)`.
    """
    views = [x] + [torch.flip(x, dims=[3] if f == 'h' else [2] if f == 'v' else [2, 3]) for f in flips]
    heat, off = model(torch.cat(views, 0))
    heat, off = torch.sigmoid(heat.float()), off.float()
    probs, offs = [heat[0:1]], [off[0:1]]
    for i, f in enumerate(flips, start=1):
        dims = [3] if f == 'h' else [2] if f == 'v' else [2, 3]
        h, o = torch.flip(heat[i:i + 1], dims=dims), torch.flip(off[i:i + 1], dims=dims).clone()
        if 3 in dims:
            o[:, 0] = 1 - o[:, 0]
        if 2 in dims:
            o[:, 1] = 1 - o[:, 1]
        probs.append(h)
        offs.append(o)
    p = torch.stack(probs).mean(0).clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p)), torch.stack(offs).mean(0)


@torch.no_grad()
def decode(heat_logits, off, stride, conf=0.1, k=10):
    """[(cls, gx, gy, score)] per batch item; gx/gy in network-input pixels."""
    hm = torch.sigmoid(heat_logits.float())
    peaks = hm * (hm == F.max_pool2d(hm, 3, 1, 1))
    B, C, H, W = hm.shape
    scores, idx = peaks.view(B, -1).topk(k)
    out = []
    for b in range(B):
        items = []
        for s, i in zip(scores[b].tolist(), idx[b].tolist()):
            if s < conf:
                break
            c, rem = divmod(i, H * W)
            y, x = divmod(rem, W)
            ox, oy = off[b, 0, y, x].item(), off[b, 1, y, x].item()
            items.append((c, (x + ox) * stride, (y + oy) * stride, s))
        out.append(items)
    return out


def letterbox_tensor(img_bgr_uint8, size, device, half):
    """numpy BGR HxWx3 -> normalised 1x3xSxS tensor on device, plus the scale r (input px -> network px)."""
    h, w = img_bgr_uint8.shape[:2]
    r = size / max(h, w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    t = torch.from_numpy(img_bgr_uint8).to(device, non_blocking=True)
    t = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255)
    t = F.interpolate(t, size=(nh, nw), mode='bilinear', align_corners=False, antialias=True)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    t = F.pad(t, (0, size - nw, 0, size - nh))
    return (t.half() if half else t), r
