"""Utilities for discovering and loading images from a folder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_files(folder_path: str | Path) -> list[Path]:
    """Return image files in a folder sorted by name."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []

    image_files = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_files, key=lambda path: path.name.lower())


def load_image(image_path: str | Path) -> np.ndarray | None:
    """Load an image as a BGR NumPy array using OpenCV."""
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        return None
    return image
