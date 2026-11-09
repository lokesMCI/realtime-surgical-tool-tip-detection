"""Inference slice: `SurgNet()` and `SurgNet.predict(image)`; the model itself is in tipnet.py."""
from .surgnet import SurgNet, WEIGHTS_DIR

__all__ = ['SurgNet', 'WEIGHTS_DIR']
