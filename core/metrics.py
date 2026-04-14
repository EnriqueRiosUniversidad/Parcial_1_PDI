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

