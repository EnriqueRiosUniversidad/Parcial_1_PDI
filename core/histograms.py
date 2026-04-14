"""Histogram utilities for grayscale image analysis."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    If the input is already grayscale, return it unchanged.
    """
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def calculate_grayscale_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """Return the 256-bin histogram of the image in grayscale."""
    grayscale_image = to_grayscale(image_bgr)
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    return histogram.flatten()

