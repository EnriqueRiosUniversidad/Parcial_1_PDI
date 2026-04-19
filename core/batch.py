"""Batch processing helpers for evaluating multiple algorithms across a folder."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.algorithms import process_algorithm
from core.image_loader import list_image_files, load_image
from core.metrics import calculate_ambe, calculate_basic_metrics, calculate_psnr


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
    batch_dir.mkdir(parents=True, exist_ok=True)

    variants = _build_variants(config)
    algorithm_totals: dict[str, dict[str, float]] = {
        variant["label"]: {"count": 0.0, "original_std": 0.0, "processed_std": 0.0, "ambe": 0.0, "psnr": 0.0}
        for variant in variants
    }
    psnr_counts: dict[str, int] = {variant["label"]: 0 for variant in variants}

    for image_path in image_files:
        image = load_image(image_path)
        if image is None:
            continue

        original_metrics = calculate_basic_metrics(image)
        for variant in variants:
            processed_image = _apply_batch_algorithm(image, variant, config)
            processed_metrics = calculate_basic_metrics(processed_image)
            ambe_value = calculate_ambe(image, processed_image)
            psnr_value = calculate_psnr(image, processed_image)

            totals = algorithm_totals[variant["label"]]
            totals["count"] += 1
            totals["original_std"] += original_metrics["desviacion_estandar"]
            totals["processed_std"] += processed_metrics["desviacion_estandar"]
            totals["ambe"] += ambe_value
            if not np.isinf(psnr_value):
                totals["psnr"] += psnr_value
                psnr_counts[variant["label"]] += 1

    comparison_csv = batch_dir / "comparativa global.csv"
    comparison_rows = _build_comparison_rows(algorithm_totals, psnr_counts)
    _write_csv(comparison_csv, comparison_rows)

    return {
        "batch_dir": batch_dir,
        "comparison_csv": comparison_csv,
    }


def build_folder_comparison(folder_path: str | Path, config: BatchConfig | None = None) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Build global and per-image comparison rows without writing files."""
    config = config or BatchConfig()
    folder = Path(folder_path)
    image_files = list_image_files(folder)
    variants = _build_variants(config)

    algorithm_totals: dict[str, dict[str, float]] = {
        variant["label"]: {"count": 0.0, "original_std": 0.0, "processed_std": 0.0, "ambe": 0.0, "psnr": 0.0}
        for variant in variants
    }
    psnr_counts: dict[str, int] = {variant["label"]: 0 for variant in variants}
    per_image_rows: dict[str, list[dict[str, object]]] = {}

    for image_path in image_files:
        image = load_image(image_path)
        if image is None:
            continue

        original_metrics = calculate_basic_metrics(image)
        image_rows: list[dict[str, object]] = []
        for variant in variants:
            processed_image = _apply_batch_algorithm(image, variant, config)
            processed_metrics = calculate_basic_metrics(processed_image)
            ambe_value = calculate_ambe(image, processed_image)
            psnr_value = calculate_psnr(image, processed_image)

            totals = algorithm_totals[variant["label"]]
            totals["count"] += 1
            totals["original_std"] += original_metrics["desviacion_estandar"]
            totals["processed_std"] += processed_metrics["desviacion_estandar"]
            totals["ambe"] += ambe_value
            if not np.isinf(psnr_value):
                totals["psnr"] += psnr_value
                psnr_counts[variant["label"]] += 1

            image_rows.append(
                {
                    "algorithm": variant["label"],
                    "original_std": round(original_metrics["desviacion_estandar"], 4),
                    "processed_std": round(processed_metrics["desviacion_estandar"], 4),
                    "ambe": round(ambe_value, 4),
                    "psnr": "Inf" if np.isinf(psnr_value) else round(psnr_value, 4),
                }
            )

        per_image_rows[image_path.name] = image_rows

    global_rows = _build_comparison_rows(algorithm_totals, psnr_counts)
    return global_rows, per_image_rows


def build_image_comparison(image_bgr: np.ndarray, config: BatchConfig | None = None) -> list[dict[str, object]]:
    """Build ranking rows for a single image across all supported algorithms."""
    config = config or BatchConfig()
    variants = _build_variants(config)
    original_metrics = calculate_basic_metrics(image_bgr)
    rows: list[dict[str, object]] = []

    for variant in variants:
        processed_image = _apply_batch_algorithm(image_bgr, variant, config)
        processed_metrics = calculate_basic_metrics(processed_image)
        ambe_value = calculate_ambe(image_bgr, processed_image)
        psnr_value = calculate_psnr(image_bgr, processed_image)
        std_delta = processed_metrics["desviacion_estandar"] - original_metrics["desviacion_estandar"]
        contrast_effect = "increase" if std_delta > 0 else "decrease" if std_delta < 0 else "neutral"
        ranking_score = (processed_metrics["desviacion_estandar"] * 0.5) + (psnr_value if not np.isinf(psnr_value) else 0.0) * 0.05 - (ambe_value * 0.5)

        rows.append(
            {
                "algorithm": variant["label"],
                "original_std": round(original_metrics["desviacion_estandar"], 4),
                "processed_std": round(processed_metrics["desviacion_estandar"], 4),
                "std_delta": round(std_delta, 4),
                "ambe": round(ambe_value, 4),
                "psnr": "Inf" if np.isinf(psnr_value) else round(psnr_value, 4),
                "contrast_effect": contrast_effect,
                "ranking_score": round(ranking_score, 4),
            }
        )

    rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def append_image_ranking_csv(csv_path: str | Path, image_name: str, rows: list[dict[str, object]]) -> None:
    """Append a single-image ranking block to a CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["category", "image_name", "rank", "algorithm", "original_std", "processed_std", "std_delta", "ambe", "psnr", "contrast_effect", "ranking_score"])

        writer.writerow(["ranking", image_name, "", "", "", "", "", "", "", "", ""])
        for row in rows:
            writer.writerow(
                [
                    "ranking",
                    image_name,
                    row["rank"],
                    row["algorithm"],
                    row["original_std"],
                    row["processed_std"],
                    row["std_delta"],
                    row["ambe"],
                    row["psnr"],
                    row["contrast_effect"],
                    row["ranking_score"],
                ]
            )
        writer.writerow([])


def _build_variants(config: BatchConfig) -> list[dict[str, object]]:
    """Create algorithm variants to evaluate in batch mode."""
    variants: list[dict[str, object]] = [
        {"algorithm": "HE", "label": "HE"},
        {"algorithm": "CLAHE", "label": f"CLAHE clip {config.clahe_clip_limit:g}", "clip_limit": config.clahe_clip_limit},
        {"algorithm": "CLAHE", "label": "CLAHE clip 1.5", "clip_limit": 1.5},
        {"algorithm": "CLAHE", "label": "CLAHE clip 3.0", "clip_limit": 3.0},
    ]

    kernel_sizes = [config.top_hat_kernel_size, 3, 5, 7, 9]
    seen_kernel_sizes: set[int] = set()
    for kernel_size in kernel_sizes:
        if kernel_size in seen_kernel_sizes:
            continue
        seen_kernel_sizes.add(kernel_size)
        variants.append({"algorithm": "White Top-Hat", "label": f"White Top-Hat {kernel_size}x{kernel_size}", "kernel_size": kernel_size})
        variants.append({"algorithm": "Black Top-Hat", "label": f"Black Top-Hat {kernel_size}x{kernel_size}", "kernel_size": kernel_size})
        variants.append({"algorithm": "Enhanced Top-Hat", "label": f"Enhanced Top-Hat {kernel_size}x{kernel_size}", "kernel_size": kernel_size})

    variants.extend(
        [
            {"algorithm": "Bilateral + CLAHE + Unsharp", "label": "Bilateral + CLAHE + Unsharp"},
            {"algorithm": "Gamma + CLAHE + Multi-scale White Top-Hat", "label": "Gamma + CLAHE + Multi-scale White Top-Hat"},
            {"algorithm": "Homomorphic + CLAHE + Enhanced Top-Hat", "label": "Homomorphic + CLAHE + Enhanced Top-Hat"},
        ]
    )

    return variants


def _apply_batch_algorithm(image_bgr: np.ndarray, variant: dict[str, object], config: BatchConfig) -> np.ndarray:
    """Apply one of the supported algorithm variants for batch processing."""
    algorithm_name = str(variant["algorithm"])
    return process_algorithm(
        image_bgr,
        algorithm_name,
        clip_limit=float(variant.get("clip_limit", config.clahe_clip_limit)),
        tile_grid_size=config.clahe_tile_grid_size,
        kernel_size=int(variant.get("kernel_size", config.top_hat_kernel_size)),
    )


def _build_comparison_rows(
    algorithm_totals: dict[str, dict[str, float]],
    psnr_counts: dict[str, int],
) -> list[dict[str, object]]:
    """Build a single global comparison table with one row per algorithm."""
    rows: list[dict[str, object]] = []

    for algorithm_name, totals in algorithm_totals.items():
        count = max(1.0, totals["count"])
        avg_original_std = totals["original_std"] / count
        avg_processed_std = totals["processed_std"] / count
        avg_ambe = totals["ambe"] / count
        avg_psnr = totals["psnr"] / max(1, psnr_counts[algorithm_name]) if psnr_counts[algorithm_name] else float("inf")
        std_delta = avg_processed_std - avg_original_std
        contrast_effect = "increase" if std_delta > 0 else "decrease" if std_delta < 0 else "neutral"
        ranking_score = (avg_processed_std * 0.5) + (avg_psnr if not np.isinf(avg_psnr) else 0.0) * 0.05 - (avg_ambe * 0.5)

        rows.append(
            {
                "algorithm": algorithm_name,
                "image_count": int(totals["count"]),
                "avg_original_std": round(avg_original_std, 4),
                "avg_processed_std": round(avg_processed_std, 4),
                "std_delta": round(std_delta, 4),
                "avg_ambe": round(avg_ambe, 4),
                "avg_psnr": "Inf" if np.isinf(avg_psnr) else round(avg_psnr, 4),
                "contrast_effect": contrast_effect,
                "ranking_score": round(ranking_score, 4),
            }
        )
    rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV if there is data."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
