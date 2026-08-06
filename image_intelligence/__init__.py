"""
Image Intelligence module for Return Fraud Detection Platform (Phase 1).
Provides schema contracts, image preprocessing, image similarity/duplicate detection,
and pure OpenCV manipulation heuristics.
"""

try:
    from .schema import (
        RiskFlag,
        ImageIntelligenceResult,
        HISTOGRAM_SIMILARITY_THRESHOLD,
        ORB_MATCH_COUNT_THRESHOLD,
        DUPLICATE_SIMILARITY_THRESHOLD,
        SCORE_WEIGHTS,
    )
    from .preprocessing import preprocess_image
    from .similarity import compare_images
    from .manipulation_check import analyze_manipulation
    from .scoring import compute_image_score, build_image_result
except ImportError:
    from schema import (
        RiskFlag,
        ImageIntelligenceResult,
        HISTOGRAM_SIMILARITY_THRESHOLD,
        ORB_MATCH_COUNT_THRESHOLD,
        DUPLICATE_SIMILARITY_THRESHOLD,
        SCORE_WEIGHTS,
    )
    from preprocessing import preprocess_image
    from similarity import compare_images
    from manipulation_check import analyze_manipulation
    from scoring import compute_image_score, build_image_result

__all__ = [
    "RiskFlag",
    "ImageIntelligenceResult",
    "HISTOGRAM_SIMILARITY_THRESHOLD",
    "ORB_MATCH_COUNT_THRESHOLD",
    "DUPLICATE_SIMILARITY_THRESHOLD",
    "SCORE_WEIGHTS",
    "preprocess_image",
    "compare_images",
    "analyze_manipulation",
    "compute_image_score",
    "build_image_result",
]
