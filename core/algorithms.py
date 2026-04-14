"""Contrast enhancement algorithms."""

from __future__ import annotations

import cv2
import numpy as np

from core.histograms import to_grayscale


def apply_histogram_equalization(image_bgr: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to the grayscale version of an image."""
    grayscale_image = to_grayscale(image_bgr)
    return cv2.equalizeHist(grayscale_image)

