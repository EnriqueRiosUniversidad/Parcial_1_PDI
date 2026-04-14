"""Morphological contrast enhancement operators."""

from __future__ import annotations

import cv2
import numpy as np

from core.histograms import to_grayscale


def _create_kernel(kernel_size: int) -> np.ndarray:
    """Create a square structuring element for morphological processing."""
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def apply_white_top_hat(image_bgr: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply the white top-hat operator to a grayscale image."""
    grayscale_image = to_grayscale(image_bgr)
    kernel = _create_kernel(kernel_size)
    opened = cv2.morphologyEx(grayscale_image, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(grayscale_image, opened)


def apply_black_top_hat(image_bgr: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply the black top-hat operator to a grayscale image."""
    grayscale_image = to_grayscale(image_bgr)
    kernel = _create_kernel(kernel_size)
    closed = cv2.morphologyEx(grayscale_image, cv2.MORPH_CLOSE, kernel)
    return cv2.subtract(closed, grayscale_image)


def apply_enhanced_top_hat(image_bgr: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply the enhanced top-hat combination IE = I + WTH - BTH."""
    grayscale_image = to_grayscale(image_bgr)
    wth = apply_white_top_hat(image_bgr, kernel_size)
    bth = apply_black_top_hat(image_bgr, kernel_size)
    enhanced = cv2.add(grayscale_image, wth)
    enhanced = cv2.subtract(enhanced, bth)
    return enhanced

