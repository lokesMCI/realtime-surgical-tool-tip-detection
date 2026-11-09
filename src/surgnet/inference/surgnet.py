"""SurgNet: the inference entry point. One BGR frame in, one tip per visible tool out.

    SurgNet()                    loads the model named in weights/config.json (TipNet, or a YOLO-pose checkpoint)
    SurgNet.predict(image_bgr)   (H, W, 3) uint8 -> list[ToolDetection(class_name, tip_xy, confidence)]
                                 tip_xy is in pixels of the input frame, origin top-left; [] when no tool is visible

The output policy (confidence per class, top-k, mirror averaging) lives in weights/config.json, so it can be changed
without touching code or retraining (see training/set_policy.py).

Needs the challenge's interface package for `ToolDetection`, `CLASSES` and `read_image`
(surgnet.shared.*), which is not part of this repository.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import numpy as np

# The graded run is offline and only /tmp is writable: keep Ultralytics from writing settings anywhere else or
# trying to reach the network.
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir()) / 'Ultralytics'))
os.environ.setdefault('YOLO_OFFLINE', '1')

from ..shared.detection import CLASSES, ToolDetection
from ..shared.images import read_image

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / 'weights'      # ship your weights here

DEFAULTS = {                                   # overridden by weights/config.json (written by training/tune_threshold.py)
    'weights': 'best.pt',
    'imgsz': 640,                              # long side the frame is resized to; keypoints are mapped back to the input
    'conf': 0.25,                              # score threshold: tuned on a by-clip validation split, not on the leaderboard
    'conf_by_class': None,                     # optional class-specific thresholds; > 1 switches a class off
    'top_k': None,                             # optional: keep only the k most confident tools per frame
    'tta': None,                               # TipNet test-time mirroring, e.g. ['h'] or ['h', 'v', 'hv']
    'iou': 0.5,
    'max_det': 10,
    'dedupe_frac': 0.02,                       # same-class detections closer than this fraction of the diagonal = one tool
    'cross_class_frac': 0.0,                   # >0: different-class tips this close = one tool seen twice, keep the surer one
    'half': True,                              # FP16 on CUDA
}


def _resolve_device(device):
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class SurgNet:
    """Loads the model once; call `predict(image)` as often as needed."""

    def __init__(self, device=None, weights=None):
        cfg = dict(DEFAULTS)
        cfg_path = WEIGHTS_DIR / 'config.json'
        if cfg_path.is_file():
            cfg.update(json.loads(cfg_path.read_text()))
        self.cfg = cfg
        self.device = _resolve_device(device)
        self.half = bool(cfg['half']) and str(self.device) not in ('cpu', 'mps')
        self._precision = {'quantize': 16} if self.half else {}          # Ultralytics >= 8.4: `half` was renamed
        path = Path(weights) if weights else WEIGHTS_DIR / cfg['weights']
        self.tipnet = path.name.endswith('.tipnet.pt')      # DINOv2 heatmap model (tipnet.py) instead of YOLO-pose
        if self.tipnet:
            self._load_tipnet(path)
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(path), task='pose')
            self.names = self.model.names                   # {id: class_name}; must be a subset of CLASSES
        self.predict(np.zeros((540, 960, 3), np.uint8))     # builds the predictor / cudnn plans outside the timed loop

    def _load_tipnet(self, path):
        import torch
        from .tipnet import TipNet
        ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
        self.model = TipNet(backbone=ckpt['backbone'], n_cls=len(ckpt['names']), size=ckpt['size'],
                            layers=ckpt['layers'], pretrained=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.torch_device = torch.device(f'cuda:{self.device}' if isinstance(self.device, int) or str(self.device).isdigit()
                                         else self.device)
        self.model.to(self.torch_device).eval()
        if self.half:
            self.model.half()
        self.names = dict(enumerate(ckpt['names']))

    def _tipnet_raw(self, image, conf):
        import torch
        from .tipnet import decode, forward_tta, letterbox_tensor
        x, r = letterbox_tensor(image, self.model.size, self.torch_device, self.half)
        with torch.inference_mode():
            heat, off = forward_tta(self.model, x, self.cfg['tta']) if self.cfg.get('tta') else self.model(x)
        h, w = image.shape[:2]
        out = []
        for c, gx, gy, s in decode(heat, off.float(), self.model.stride, conf=conf, k=self.cfg['max_det'])[0]:
            out.append((c, min(max(gx / r, 0.0), w - 1.0), min(max(gy / r, 0.0), h - 1.0), s))
        return out

    def predict_raw(self, image: np.ndarray, conf: float | None = None) -> list[ToolDetection]:
        """All detections above `conf` (default: the tuned threshold) before de-duplication. Used for threshold tuning."""
        conf_by_class = self.cfg.get('conf_by_class') or {}
        model_conf = (min([self.cfg['conf'], *conf_by_class.values()])
                      if conf is None and conf_by_class else
                      (self.cfg['conf'] if conf is None else conf))
        if self.tipnet:
            out = []
            for c, x, y, s in self._tipnet_raw(image, model_conf):
                name = self.names[int(c)]
                threshold = conf_by_class.get(name, self.cfg['conf']) if conf is None else conf
                if name in CLASSES and s >= threshold and np.isfinite([x, y]).all():
                    out.append(ToolDetection(name, (float(x), float(y)), float(min(max(s, 0.0), 1.0))))
            return out
        r = self.model.predict(image, imgsz=self.cfg['imgsz'], conf=model_conf,
                               iou=self.cfg['iou'], max_det=self.cfg['max_det'], device=self.device,
                               verbose=False, **self._precision)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []
        scores = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        xy = r.keypoints.xy.cpu().numpy()[:, 0, :]          # (n, 2) tip in input-image pixels
        out = []
        for s, c, p in zip(scores, cls, xy):
            name = self.names[int(c)]
            threshold = conf_by_class.get(name, self.cfg['conf']) if conf is None else conf
            if name in CLASSES and s >= threshold and np.isfinite(p).all():
                out.append(ToolDetection(name, (float(p[0]), float(p[1])), float(min(max(s, 0.0), 1.0))))
        return out

    def _dedupe(self, dets: list[ToolDetection], diag: float) -> list[ToolDetection]:
        """Two boxes on one tool give two tips a few px apart: the second would be a false positive."""
        radius = self.cfg['dedupe_frac'] * diag
        cross = self.cfg.get('cross_class_frac', 0.0) * diag
        kept: list[ToolDetection] = []
        for d in sorted(dets, key=lambda d: -d.confidence):
            if all(np.hypot(k.tip_xy[0] - d.tip_xy[0], k.tip_xy[1] - d.tip_xy[1]) > (radius if k.class_name == d.class_name
                                                                                    else cross)
                   for k in kept):
                kept.append(d)
        return kept

    def predict(self, image: np.ndarray) -> list[ToolDetection]:
        """Detect every tool in one BGR uint8 frame of shape (H, W, 3).

        Returns a list of ToolDetection(class_name, tip_xy, confidence), one per tool, in pixel coordinates of
        `image` (origin top-left). Return [] when no tool is visible. See the module docstring for the full rules.
        """
        h, w = image.shape[:2]
        dets = self._dedupe(self.predict_raw(image), float(np.hypot(w, h)))
        if self.cfg.get('top_k'):
            dets = sorted(dets, key=lambda d: -d.confidence)[:int(self.cfg['top_k'])]
        return dets

    def predict_path(self, image_path) -> list[ToolDetection]:
        """Convenience: read the image from disk, then `predict`. The benchmark does not use this."""
        return self.predict(read_image(image_path))
