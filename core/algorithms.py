"""Contrast enhancement algorithms."""

from __future__ import annotations

import cv2
import numpy as np

from core.histograms import to_grayscale
from core.morphology import apply_black_top_hat, apply_enhanced_top_hat, apply_white_top_hat


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Clamp and convert an image to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize a float image to the 0-255 range and convert to uint8."""
    normalized = np.empty_like(image, dtype=np.float32)
    cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
    return _to_uint8(normalized)


def apply_histogram_equalization(image_bgr: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to the grayscale version of an image."""
    grayscale_image = to_grayscale(image_bgr)
    return cv2.equalizeHist(grayscale_image)


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to the grayscale version of an image."""
    grayscale_image = to_grayscale(image_bgr)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(grayscale_image)


def bilateral_filter(image_bgr: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Reduce noise while preserving edges using bilateral filtering."""
    grayscale_image = to_grayscale(image_bgr)
    return cv2.bilateralFilter(grayscale_image, d=int(d), sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


def gamma_correction(image_bgr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply power-law intensity transformation with safe normalization."""
    grayscale_image = to_grayscale(image_bgr).astype(np.float32)
    if grayscale_image.size == 0:
        return np.zeros_like(grayscale_image, dtype=np.uint8)

    normalized = np.clip(grayscale_image / 255.0, 0.0, 1.0)
    gamma_value = max(float(gamma), 1e-6)
    corrected = np.power(np.clip(normalized, 0.0, 1.0), gamma_value)
    corrected = corrected * 255.0
    return _to_uint8(corrected)


def unsharp_mask(image_bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.5, threshold: float = 0.0) -> np.ndarray:
    """Sharpen edges with a controlled unsharp mask."""
    grayscale_image = to_grayscale(image_bgr)
    blurred = cv2.GaussianBlur(grayscale_image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharpened = cv2.addWeighted(grayscale_image, float(amount), blurred, -float(amount - 1.0), float(0.0))
    if threshold > 0:
        low_contrast_mask = np.abs(grayscale_image.astype(np.int16) - blurred.astype(np.int16)) < float(threshold)
        sharpened = np.where(low_contrast_mask, grayscale_image, sharpened)
    return _to_uint8(sharpened)


def homomorphic_filter(
    image_bgr: np.ndarray,
    gamma_l: float = 0.5,
    gamma_h: float = 1.8,
    c: float = 1.0,
    d0: float = 30.0,
) -> np.ndarray:
    """Apply a basic homomorphic enhancement in the frequency domain."""
    grayscale_image = to_grayscale(image_bgr).astype(np.float32)
    if grayscale_image.size == 0:
        return np.zeros_like(grayscale_image, dtype=np.uint8)

    image_log = np.log1p(grayscale_image)
    rows, cols = image_log.shape
    d0_value = max(float(d0), 1.0)

    y, x = np.ogrid[:rows, :cols]
    center_y = rows / 2.0
    center_x = cols / 2.0
    distance_squared = (y - center_y) ** 2 + (x - center_x) ** 2
    transfer = (float(gamma_h) - float(gamma_l)) * (1.0 - np.exp(-float(c) * distance_squared / (d0_value * d0_value))) + float(gamma_l)

    frequency = np.fft.fftshift(np.fft.fft2(image_log))
    filtered_frequency = frequency * transfer
    reconstructed = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_frequency)))
    enhanced = np.expm1(reconstructed)
    return _normalize_to_uint8(enhanced.astype(np.float32))


def multi_scale_white_tophat(image_bgr: np.ndarray, kernel_sizes: tuple[int, ...] = (3, 5, 9, 15)) -> np.ndarray:
    """Combine white top-hat responses from several kernel sizes."""
    grayscale_image = to_grayscale(image_bgr)
    responses: list[np.ndarray] = []

    for kernel_size in kernel_sizes:
        response = apply_white_top_hat(grayscale_image, int(kernel_size))
        responses.append(response.astype(np.float32))

    if not responses:
        return grayscale_image.copy()

    stacked = np.stack(responses, axis=0)
    combined = np.mean(stacked, axis=0)
    return _normalize_to_uint8(combined)


def pipeline_bilateral_clahe_unsharp(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply bilateral filtering, CLAHE and unsharp masking in sequence."""
    filtered = bilateral_filter(image_bgr)
    equalized = apply_clahe(filtered, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return unsharp_mask(equalized)


def pipeline_gamma_clahe_multi_tophat(
    image_bgr: np.ndarray,
    gamma: float = 1.0,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply gamma correction, CLAHE and multi-scale white top-hat in sequence."""
    corrected = gamma_correction(image_bgr, gamma=gamma)
    equalized = apply_clahe(corrected, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return multi_scale_white_tophat(equalized)


def pipeline_homomorphic_clahe_enhanced_tophat(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    kernel_size: int = 15,
) -> np.ndarray:
    """Apply homomorphic filtering, CLAHE and enhanced top-hat in sequence."""
    filtered = homomorphic_filter(image_bgr)
    equalized = apply_clahe(filtered, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return apply_enhanced_top_hat(equalized, kernel_size=kernel_size)


def apply_morphological_algorithm(image_bgr: np.ndarray, algorithm_name: str, kernel_size: int = 15) -> np.ndarray:
    """Dispatch morphological contrast algorithms."""
    if algorithm_name == "White Top-Hat":
        return apply_white_top_hat(image_bgr, kernel_size)
    if algorithm_name == "Black Top-Hat":
        return apply_black_top_hat(image_bgr, kernel_size)
    if algorithm_name == "Enhanced Top-Hat":
        return apply_enhanced_top_hat(image_bgr, kernel_size)
    return to_grayscale(image_bgr)


def process_algorithm(
    image_bgr: np.ndarray,
    algorithm_name: str,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    kernel_size: int = 15,
) -> np.ndarray:
    """Apply one supported algorithm or pipeline by its display name."""
    if algorithm_name == "HE":
        return apply_histogram_equalization(image_bgr)
    if algorithm_name == "CLAHE":
        return apply_clahe(image_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    if algorithm_name in {"White Top-Hat", "Black Top-Hat", "Enhanced Top-Hat"}:
        return apply_morphological_algorithm(image_bgr, algorithm_name, kernel_size=kernel_size)
    if algorithm_name == "Bilateral + CLAHE + Unsharp":
        return pipeline_bilateral_clahe_unsharp(image_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    if algorithm_name == "Gamma + CLAHE + Multi-scale White Top-Hat":
        return pipeline_gamma_clahe_multi_tophat(image_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    if algorithm_name == "Homomorphic + CLAHE + Enhanced Top-Hat":
        return pipeline_homomorphic_clahe_enhanced_tophat(
            image_bgr,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            kernel_size=kernel_size,
        )
    return apply_histogram_equalization(image_bgr)
