"""
similarity.py

Detects duplicate or reused product damage images across return claims using
OpenCV HSV color histogram correlation and ORB (Oriented FAST and Rotated BRIEF)
feature keypoint matching.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2

from schema import (
    HISTOGRAM_SIMILARITY_THRESHOLD,
    ORB_MATCH_COUNT_THRESHOLD,
    DUPLICATE_SIMILARITY_THRESHOLD,
)


def compute_histogram_similarity(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    """
    Computes HSV color space histogram correlation between two BGR images.
    Returns correlation value in [-1.0, 1.0], normalized to [0.0, 1.0].
    """
    hsv1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)

    # 2D H-S histogram
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(max(0.0, correlation))


def compute_orb_similarity(
    gray1: np.ndarray, gray2: np.ndarray, max_features: int = 500
) -> Tuple[int, float, float]:
    """
    Extracts ORB feature descriptors and performs Hamming distance matching.

    Returns:
        (good_matches_count, good_match_ratio, orb_score)
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0, 0.0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter good matches (Hamming distance < 50)
    good_matches = [m for m in matches if m.distance < 50.0]
    good_count = len(good_matches)

    min_kp = min(len(kp1), len(kp2))
    good_ratio = (good_count / min_kp) if min_kp > 0 else 0.0

    # Normalize ORB match score relative to threshold ceiling
    orb_score = min(good_count / ORB_MATCH_COUNT_THRESHOLD, 1.0)

    return good_count, round(good_ratio, 4), round(orb_score, 4)


def compare_images(
    img_data1: Dict[str, Any], img_data2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compares two preprocessed image dictionaries for image reuse / duplicate detection.

    Returns:
        dict with histogram correlation, ORB match metrics, and combined similarity.
    """
    hist_sim = compute_histogram_similarity(img_data1["bgr"], img_data2["bgr"])
    orb_count, orb_ratio, orb_score = compute_orb_similarity(
        img_data1["gray"], img_data2["gray"]
    )

    # Combined similarity score: weighted blend of color histogram and feature matches
    combined_similarity = 0.40 * hist_sim + 0.60 * orb_score

    is_exact_duplicate = (hist_sim >= 0.95) and (orb_count >= ORB_MATCH_COUNT_THRESHOLD)
    is_near_duplicate = (combined_similarity >= DUPLICATE_SIMILARITY_THRESHOLD)

    return {
        "histogram_correlation": round(hist_sim, 4),
        "orb_matches": orb_count,
        "orb_match_ratio": orb_ratio,
        "orb_score": orb_score,
        "combined_similarity": round(combined_similarity, 4),
        "is_exact_duplicate": is_exact_duplicate,
        "is_near_duplicate": is_near_duplicate,
    }
