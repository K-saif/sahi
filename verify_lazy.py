
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

try:
    import sahi  # noqa: F401
except ImportError:
    # run from a checkout without installing it
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from sahi.slicing import slice_image
from sahi.utils.cv import read_image_as_pil, read_image_size
from sahi.utils.lazy_image import is_lazy_image_source
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction




detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/home/medprime/Downloads/agg_giant_best.pt",
    confidence_threshold=0.25,
    device="cpu",  # or "cuda:0"
)

result = get_sliced_prediction(
    "/home/medprime/Desktop/tissue_analysis/tissue_tiff/1695811378215-A1.tiff",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)