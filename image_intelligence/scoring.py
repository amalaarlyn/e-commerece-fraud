"""
scoring.py

Combines image similarity metrics (HSV histogram + ORB feature keypoint matching)
and pure OpenCV manipulation heuristics (edge, noise, ELA anomalies) into a single
normalized image_score (0-1), and generates human-readable risk_flags for explainability
(consumed by Member 3's Risk Fusion Engine and frontend dashboard).

Mirroring graph_intelligence/scoring.py's design:
  - Clear, transparent weighted scoring formula.
  - Explainable RiskFlag list with machine-readable codes and severity metrics.
  - Upgrade path to Phase 2 deep learning (CLIP/ResNet/ELA CNN) without altering
    the schema contract.
"""

from typing import Dict, Any, List, Optional
from schema import (
    SCORE_WEIGHTS,
    EXACT_DUPLICATE_FLOOR,
    NEAR_DUPLICATE_FLOOR,
    MANIPULATION_ONLY_CAP,
    RiskFlag,
    ImageIntelligenceResult,
    HISTOGRAM_SIMILARITY_THRESHOLD,
    ORB_MATCH_COUNT_THRESHOLD,
    DUPLICATE_SIMILARITY_THRESHOLD,
)
from similarity import compare_images
from manipulation_check import analyze_manipulation


def compute_image_score(
    uploaded_img_data: Dict[str, Any],
    candidate_gallery: Optional[List[Dict[str, Any]]] = None,
    customer_id: str = "CUST_DEFAULT",
    return_id: Optional[str] = None,
    image_id: Optional[str] = None,
) -> ImageIntelligenceResult:
    """
    Evaluates an uploaded image against a candidate gallery of previously submitted
    claim images and executes digital manipulation checks.

    Args:
        uploaded_img_data: Preprocessed dictionary from preprocess_image()
        candidate_gallery: List of dicts, each having 'img_data', 'customer_id', 'claim_id', 'image_id'
        customer_id: Top-level customer identifier
        return_id: Return claim identifier
        image_id: Image file identifier

    Returns:
        ImageIntelligenceResult dataclass instance
    """
    # 1. Analyze digital manipulation heuristics
    manipulation_res = analyze_manipulation(uploaded_img_data)
    manipulation_score = manipulation_res["overall_manipulation_score"]

    # 2. Compare against candidate gallery (duplicate / image reuse detection)
    max_sim_score = 0.0
    max_hist_sim = 0.0
    max_orb_matches = 0
    matched_claim_id = None
    matched_customer_id = None
    best_match_details = {}

    if candidate_gallery:
        for cand in candidate_gallery:
            sim = compare_images(uploaded_img_data, cand["img_data"])
            if sim["combined_similarity"] > max_sim_score:
                max_sim_score = sim["combined_similarity"]
                max_hist_sim = sim["histogram_correlation"]
                max_orb_matches = sim["orb_matches"]
                matched_claim_id = cand.get("return_id") or cand.get("claim_id")
                matched_customer_id = cand.get("customer_id")
                best_match_details = sim

    # 3. Calculate normalized final image_score (0.0 to 1.0) with duplicate floors & manipulation cap
    raw_score = (
        SCORE_WEIGHTS["similarity"] * max_sim_score
        + SCORE_WEIGHTS["manipulation"] * manipulation_score
    )

    if best_match_details.get("is_exact_duplicate"):
        # Ground-truth exact duplicate evidence pushes score to ceiling [0.95, 1.0]
        final_score = max(raw_score, EXACT_DUPLICATE_FLOOR)
    elif best_match_details.get("is_near_duplicate") or max_sim_score >= DUPLICATE_SIMILARITY_THRESHOLD:
        # Near duplicate image reuse gets floor of 0.80, strictly capped below exact duplicate floor
        final_score = min(max(raw_score, NEAR_DUPLICATE_FLOOR), EXACT_DUPLICATE_FLOOR - 0.01)
    else:
        # Manipulation heuristics alone (probabilistic) reflect manipulation_score capped at MANIPULATION_ONLY_CAP (0.75)
        final_score = min(manipulation_score, MANIPULATION_ONLY_CAP)

    final_score = min(max(0.0, final_score), 1.0)

    # 4. Build explainable RiskFlags (Front & center in output schema)
    flags: List[RiskFlag] = []

    # Similarity risk flags
    if best_match_details.get("is_exact_duplicate"):
        flags.append(
            RiskFlag(
                code="EXACT_DUPLICATE_IMAGE",
                message=f"Exact duplicate image detected (matches return claim '{matched_claim_id}' from customer '{matched_customer_id}')",
                severity=1.0,
            )
        )
    elif max_sim_score >= DUPLICATE_SIMILARITY_THRESHOLD:
        flags.append(
            RiskFlag(
                code="NEAR_DUPLICATE_IMAGE",
                message=f"High similarity ({round(max_sim_score*100, 1)}%) detected with previously submitted claim image '{matched_claim_id}'",
                severity=round(max_sim_score, 2),
            )
        )

    if max_orb_matches >= ORB_MATCH_COUNT_THRESHOLD:
        flags.append(
            RiskFlag(
                code="FEATURE_MATCH_REUSE",
                message=f"Reused visual keypoints detected ({max_orb_matches} matching ORB features found across claims)",
                severity=round(min(max_orb_matches / 40.0, 1.0), 2),
            )
        )

    if max_hist_sim >= HISTOGRAM_SIMILARITY_THRESHOLD and not best_match_details.get("is_exact_duplicate"):
        flags.append(
            RiskFlag(
                code="HIGH_HISTOGRAM_MATCH",
                message=f"Color distribution match of {round(max_hist_sim*100, 1)}% with existing return claim photo",
                severity=round(max_hist_sim, 2),
            )
        )

    # Manipulation risk flags (only raised when heuristic score > 0.0)
    edge_score = manipulation_res["edge_analysis"]["edge_inconsistency_score"]
    if edge_score > 0.0:
        flags.append(
            RiskFlag(
                code="SPLICING_EDGE_ANOMALY",
                message=f"Unnatural spatial edge disparity detected (sharpness ratio {manipulation_res['edge_analysis']['edge_disparity_ratio']}x indicates potential cutout splicing)",
                severity=edge_score,
            )
        )

    noise_score = manipulation_res["noise_analysis"]["noise_inconsistency_score"]
    if noise_score > 0.0:
        flags.append(
            RiskFlag(
                code="UNNATURAL_NOISE_VARIANCE",
                message=f"Non-uniform spatial noise variance detected (noise coef variation {manipulation_res['noise_analysis']['noise_coef_variation']} suggests pasted regions)",
                severity=noise_score,
            )
        )

    ela_score = manipulation_res["ela_analysis"]["ela_discrepancy_score"]
    if ela_score > 0.0:
        flags.append(
            RiskFlag(
                code="COMPRESSION_ARTIFACT_DISCREPANCY",
                message="Error Level Analysis (ELA) discrepancy detected (localized post-compression edit boundaries)",
                severity=ela_score,
            )
        )

    # Assemble structured summary details
    similarity_summary = {
        "max_similarity_score": round(max_sim_score, 4),
        "max_histogram_correlation": max_hist_sim,
        "max_orb_matches": max_orb_matches,
        "matched_customer_id": matched_customer_id,
        "matched_return_id": matched_claim_id,
        "details": best_match_details,
    }

    return ImageIntelligenceResult(
        customer_id=customer_id,
        image_score=round(final_score, 4),
        risk_flags=flags,
        return_id=return_id,
        image_id=image_id,
        similarity_details=similarity_summary,
        manipulation_details=manipulation_res,
    )


def build_image_result(
    customer_id: str,
    uploaded_img_data: Dict[str, Any],
    candidate_gallery: Optional[List[Dict[str, Any]]] = None,
    return_id: Optional[str] = None,
    image_id: Optional[str] = None,
) -> ImageIntelligenceResult:
    """
    Convenience wrapper to compute score and assemble ImageIntelligenceResult.
    """
    return compute_image_score(
        uploaded_img_data=uploaded_img_data,
        candidate_gallery=candidate_gallery,
        customer_id=customer_id,
        return_id=return_id,
        image_id=image_id,
    )
