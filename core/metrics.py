"""Basic image metrics for grayscale analysis."""

from __future__ import annotations

import numpy as np

from core.histograms import to_grayscale


def calculate_basic_metrics(image_bgr: np.ndarray) -> dict[str, float]:
    """Compute base statistics for a grayscale version of the image."""
    grayscale_image = to_grayscale(image_bgr)
    pixel_values = grayscale_image.astype(np.float32)

    return {
        "media": float(np.mean(pixel_values)),
        "desviacion_estandar": float(np.std(pixel_values)),
    }


def calculate_ambe(original_image: np.ndarray, processed_image: np.ndarray) -> float:
    """Calculate the absolute mean brightness error between two images."""
    original_gray = to_grayscale(original_image).astype(np.float32)
    processed_gray = to_grayscale(processed_image).astype(np.float32)
    return float(abs(np.mean(original_gray) - np.mean(processed_gray)))


def calculate_psnr(original_image: np.ndarray, processed_image: np.ndarray) -> float:
    """Calculate the peak signal-to-noise ratio between two images."""
    original_gray = to_grayscale(original_image).astype(np.float32)
    processed_gray = to_grayscale(processed_image).astype(np.float32)

    mse = float(np.mean((original_gray - processed_gray) ** 2))
    if mse == 0:
        return float("inf")

    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def evaluate_contrast_change(original_std: float, processed_std: float) -> str:
    """Return a short text describing whether contrast improved."""
    if processed_std > original_std:
        return "El contraste aumentó."
    if processed_std < original_std:
        return "El contraste disminuyó."
    return "El contraste se mantuvo."


def evaluate_brightness_preservation(ambe: float, threshold: float = 10.0) -> str:
    """Return a short text describing whether brightness was preserved reasonably."""
    if ambe <= threshold:
        return "El brillo se preservó razonablemente."
    return "El brillo cambió de forma notable."
