"""Contrast enhancement algorithms."""

from __future__ import annotations

import cv2
import numpy as np

from core.histograms import to_grayscale
from core.morphology import apply_black_top_hat, apply_enhanced_top_hat, apply_white_top_hat


def apply_histogram_equalization(image_bgr: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to the grayscale version of an image."""
    grayscale_image = to_grayscale(image_bgr)
    return cv2.equalizeHist(grayscale_image)


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to the grayscale version of an image."""
    grayscale_image = to_grayscale(image_bgr)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(grayscale_image)


def apply_morphological_algorithm(image_bgr: np.ndarray, algorithm_name: str, kernel_size: int = 15) -> np.ndarray:
    """Dispatch morphological contrast algorithms."""
    if algorithm_name == "White Top-Hat":
        return apply_white_top_hat(image_bgr, kernel_size)
    if algorithm_name == "Black Top-Hat":
        return apply_black_top_hat(image_bgr, kernel_size)
    if algorithm_name == "Enhanced Top-Hat":
        return apply_enhanced_top_hat(image_bgr, kernel_size)
    return to_grayscale(image_bgr)
