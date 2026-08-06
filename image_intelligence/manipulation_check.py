"""
manipulation_check.py

Pure OpenCV digital image manipulation heuristics (Phase 1 zero-ML tier):
  1. Spatial Laplacian edge inconsistency (detects unnatural sharpness/blur boundaries).
  2. Local grid noise variance inconsistency (detects spliced patches with non-uniform sensor noise).
  3. JPEG Error Level Analysis (ELA) discrepancy (detects post-compression edited regions).
"""

from typing import Dict, Any
import numpy as np
import cv2


def check_edge_inconsistency(gray: np.ndarray, grid_size: int = 4) -> Dict[str, float]:
    """
    Evaluates spatial edge variance across a grid of image patches.
    Spliced/cutout elements create sharp edge variance spikes relative to surroundings.
    """
    h, w = gray.shape
    bh, bw = h // grid_size, w // grid_size

    variances = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = gray[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            variances.append(float(lap.var()))

    max_var = max(variances) if variances else 0.0
    median_var = float(np.median(variances)) if variances else 1.0

    # Floor median_var at 50.0 to prevent division spikes on smooth background patches
    denom = max(median_var, 50.0)
    ratio = max_var / denom

    # Unnatural edge disparity requires both high disparity ratio (>12x) AND high absolute sharpness (>1500)
    if max_var > 1500.0 and ratio > 12.0:
        score = min(max(0.0, ratio - 12.0) / 20.0, 1.0)
    else:
        score = 0.0

    return {
        "edge_max_variance": round(max_var, 2),
        "edge_median_variance": round(median_var, 2),
        "edge_disparity_ratio": round(ratio, 2),
        "edge_inconsistency_score": round(score, 4),
    }


def check_noise_inconsistency(gray: np.ndarray, grid_size: int = 4) -> Dict[str, float]:
    """
    Measures local high-frequency noise variance across spatial tiles.
    Pasted patches carry different sensor noise profiles from the base image.
    """
    # High-pass noise residual via Gaussian blur subtraction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = cv2.absdiff(gray, blurred)

    h, w = gray.shape
    bh, bw = h // grid_size, w // grid_size

    tile_noises = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile = noise_residual[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            tile_noises.append(float(tile.var()))

    mean_noise = float(np.mean(tile_noises)) if tile_noises else 0.0
    std_noise = float(np.std(tile_noises)) if tile_noises else 0.0

    cv_noise = (std_noise / (mean_noise + 1e-5)) if mean_noise > 0 else 0.0

    # Low noise variation (std_noise < 25.0) is normal sensor/texture noise.
    # Non-uniform noise splicing requires std_noise >= 25.0 AND cv_noise > 0.85
    if std_noise >= 25.0 and cv_noise > 0.85:
        score = min(max(0.0, cv_noise - 0.85) / 0.50, 1.0)
    else:
        score = 0.0

    return {
        "noise_mean_variance": round(mean_noise, 2),
        "noise_std_variance": round(std_noise, 2),
        "noise_coef_variation": round(cv_noise, 4),
        "noise_inconsistency_score": round(score, 4),
    }


def check_ela_discrepancy(bgr: np.ndarray, quality: int = 90) -> Dict[str, float]:
    """
    Error Level Analysis (ELA): Re-compresses image to JPEG in memory and measures
    spatial error delta variance to locate edited regions.
    """
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return {"ela_mean_delta": 0.0, "ela_std_delta": 0.0, "ela_discrepancy_score": 0.0}

    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    ela_map = cv2.absdiff(bgr, recompressed)
    ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)

    scale = 15.0
    scaled_ela = cv2.convertScaleAbs(ela_gray, alpha=scale)

    mean_delta = float(np.mean(scaled_ela))
    std_delta = float(np.std(scaled_ela))

    # Standard JPEG quantization produces std_delta up to ~35.0 across edges.
    # Anomaly score triggers when std_delta > 35.0
    if std_delta > 35.0:
        score = min(max(0.0, std_delta - 35.0) / 25.0, 1.0)
    else:
        score = 0.0

    return {
        "ela_mean_delta": round(mean_delta, 2),
        "ela_std_delta": round(std_delta, 2),
        "ela_discrepancy_score": round(score, 4),
    }


def analyze_manipulation(img_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregates all digital manipulation heuristics into a summary dict.
    """
    edge_res = check_edge_inconsistency(img_data["gray"])
    noise_res = check_noise_inconsistency(img_data["gray"])
    ela_res = check_ela_discrepancy(img_data["bgr"])

    # Overall manipulation anomaly score (max / combined metric)
    overall_manipulation_score = round(
        max(
            edge_res["edge_inconsistency_score"],
            noise_res["noise_inconsistency_score"],
            ela_res["ela_discrepancy_score"],
        ),
        4,
    )

    return {
        "edge_analysis": edge_res,
        "noise_analysis": noise_res,
        "ela_analysis": ela_res,
        "overall_manipulation_score": overall_manipulation_score,
    }
