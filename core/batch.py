"""Batch processing helpers for evaluating multiple algorithms across a folder."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.algorithms import apply_clahe, apply_histogram_equalization, apply_morphological_algorithm
from core.image_loader import list_image_files, load_image
from core.metrics import calculate_ambe, calculate_basic_metrics, calculate_psnr


ALGORITHM_NAMES = [
    "HE",
    "CLAHE",
    "White Top-Hat",
    "Black Top-Hat",
    "Enhanced Top-Hat",
]


@dataclass
class BatchConfig:
    """Batch execution parameters."""

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    top_hat_kernel_size: int = 15


def run_folder_batch(folder_path: str | Path, output_root: str | Path, config: BatchConfig | None = None) -> dict[str, Path]:
    """Process all JPG images in a folder with all supported algorithms."""
    config = config or BatchConfig()
    folder = Path(folder_path)
    output_root = Path(output_root)
    image_files = list_image_files(folder)

    batch_dir = output_root / f"{folder.name}_batch"
    processed_dir = batch_dir / "processed_images"
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    algorithm_totals: dict[str, dict[str, float]] = {
        name: {"count": 0.0, "original_std": 0.0, "processed_std": 0.0, "ambe": 0.0, "psnr": 0.0}
        for name in ALGORITHM_NAMES
    }
    psnr_counts: dict[str, int] = {name: 0 for name in ALGORITHM_NAMES}

    for image_path in image_files:
        image = load_image(image_path)
        if image is None:
            continue

        original_metrics = calculate_basic_metrics(image)
        for algorithm_name in ALGORITHM_NAMES:
            processed_image = _apply_batch_algorithm(image, algorithm_name, config)
            processed_metrics = calculate_basic_metrics(processed_image)
            ambe_value = calculate_ambe(image, processed_image)
            psnr_value = calculate_psnr(image, processed_image)

            algorithm_dir = processed_dir / algorithm_name.replace(" ", "_")
            algorithm_dir.mkdir(parents=True, exist_ok=True)
            output_image_path = algorithm_dir / image_path.name
            cv2.imwrite(str(output_image_path), processed_image)

            summary_rows.append(
                {
                    "image_name": image_path.name,
                    "algorithm": algorithm_name,
                    "original_std": round(original_metrics["desviacion_estandar"], 4),
                    "processed_std": round(processed_metrics["desviacion_estandar"], 4),
                    "ambe": round(ambe_value, 4),
                    "psnr": "Inf" if np.isinf(psnr_value) else round(psnr_value, 4),
                    "output_image": str(output_image_path),
                }
            )

            totals = algorithm_totals[algorithm_name]
            totals["count"] += 1
            totals["original_std"] += original_metrics["desviacion_estandar"]
            totals["processed_std"] += processed_metrics["desviacion_estandar"]
            totals["ambe"] += ambe_value
            if not np.isinf(psnr_value):
                totals["psnr"] += psnr_value
                psnr_counts[algorithm_name] += 1

    per_image_csv = batch_dir / "batch_summary_by_image.csv"
    global_csv = batch_dir / "batch_global_comparison.csv"
    ranking_csv = batch_dir / "batch_algorithm_ranking.csv"

    _write_csv(per_image_csv, summary_rows)
    ranking_rows = _build_ranking_rows(algorithm_totals, psnr_counts)
    _write_csv(global_csv, ranking_rows["global_rows"])
    _write_csv(ranking_csv, ranking_rows["ranking_rows"])

    return {
        "batch_dir": batch_dir,
        "per_image_csv": per_image_csv,
        "global_csv": global_csv,
        "ranking_csv": ranking_csv,
        "processed_dir": processed_dir,
    }


def _apply_batch_algorithm(image_bgr: np.ndarray, algorithm_name: str, config: BatchConfig) -> np.ndarray:
    """Apply one of the supported algorithms for batch processing."""
    if algorithm_name == "HE":
        return apply_histogram_equalization(image_bgr)
    if algorithm_name == "CLAHE":
        return apply_clahe(
            image_bgr,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )
    if algorithm_name in {"White Top-Hat", "Black Top-Hat", "Enhanced Top-Hat"}:
        return apply_morphological_algorithm(image_bgr, algorithm_name, kernel_size=config.top_hat_kernel_size)
    return apply_histogram_equalization(image_bgr)


def _build_ranking_rows(
    algorithm_totals: dict[str, dict[str, float]],
    psnr_counts: dict[str, int],
) -> dict[str, list[dict[str, object]]]:
    """Build global comparison and ranking tables."""
    global_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    for algorithm_name, totals in algorithm_totals.items():
        count = max(1.0, totals["count"])
        psnr_count = max(1, psnr_counts[algorithm_name])
        avg_original_std = totals["original_std"] / count
        avg_processed_std = totals["processed_std"] / count
        avg_ambe = totals["ambe"] / count
        avg_psnr = totals["psnr"] / psnr_count if psnr_counts[algorithm_name] else float("inf")

        global_rows.append(
            {
                "algorithm": algorithm_name,
                "image_count": int(totals["count"]),
                "avg_original_std": round(avg_original_std, 4),
                "avg_processed_std": round(avg_processed_std, 4),
                "avg_ambe": round(avg_ambe, 4),
                "avg_psnr": "Inf" if np.isinf(avg_psnr) else round(avg_psnr, 4),
            }
        )

        ranking_score = avg_processed_std - avg_ambe
        ranking_rows.append(
            {
                "algorithm": algorithm_name,
                "avg_processed_std": round(avg_processed_std, 4),
                "avg_ambe": round(avg_ambe, 4),
                "avg_psnr": "Inf" if np.isinf(avg_psnr) else round(avg_psnr, 4),
                "ranking_score": round(ranking_score, 4),
            }
        )

    ranking_rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    return {"global_rows": global_rows, "ranking_rows": ranking_rows}


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV if there is data."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

