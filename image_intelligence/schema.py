"""
schema.py

Central place for:
  - Configuration thresholds (histogram similarity, ORB match count, duplicate detection)
  - Scoring weights (similarity vs. digital manipulation heuristics)
  - Typed output contracts (ImageIntelligenceResult and RiskFlag handed to
    Member 3's Risk Fusion Engine and to the frontend via the backend API)

Keeping this separate from algorithm code ensures:
  1. Thresholds are tunable in one place without touching vision/similarity logic.
  2. When upgrading to deep learning models (e.g. CLIP / ResNet / ELA CNN) in Phase 2,
     the output schema contract remains unchanged.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Config & Thresholds -- Image similarity & manipulation triggers
# ---------------------------------------------------------------------------
HISTOGRAM_SIMILARITY_THRESHOLD = 0.85     # HSV histogram correlation threshold
ORB_MATCH_COUNT_THRESHOLD = 25            # Minimum ORB matched feature keypoints
DUPLICATE_SIMILARITY_THRESHOLD = 0.75     # Combined similarity confidence threshold

# ---------------------------------------------------------------------------
# Scoring weights & risk floors -- how much similarity vs manipulation checks contribute
# to the final image_score (0-1), with ground-truth duplicate evidence prioritized.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "similarity": 0.70,       # Reused / duplicate image across claims (primary signal)
    "manipulation": 0.30,     # Digital tampering / edge / noise anomalies (secondary signal)
}

EXACT_DUPLICATE_FLOOR = 0.95   # Exact pixel/feature duplicate floor
NEAR_DUPLICATE_FLOOR = 0.80    # Near duplicate (cropped/edited reuse) floor
MANIPULATION_ONLY_CAP = 0.75   # Standalone manipulation score cap (without duplicate match)


# ---------------------------------------------------------------------------
# Output contracts -- Exact mirror of GraphIntelligenceResult shape
# ---------------------------------------------------------------------------
@dataclass
class RiskFlag:
    code: str          # machine-readable, e.g. "DUPLICATE_IMAGE_DETECTED"
    message: str       # human-readable explanation for dashboard
    severity: float    # 0-1, how much this flag alone contributes to suspicion


@dataclass
class ImageIntelligenceResult:
    customer_id: str                          # Top-level identifier matching GraphIntelligenceResult
    image_score: float                        # 0-1, final normalized score
    risk_flags: List[RiskFlag]                # List of human-readable explainable risk flags
    return_id: Optional[str] = None           # Return claim ID context
    image_id: Optional[str] = None            # Specific image identifier within the claim
    similarity_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    manipulation_details: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes to the exact JSON contract handed to Member 3's Risk Fusion Engine
        and the frontend dashboard.
        """
        return {
            "customer_id": self.customer_id,
            "image_score": round(self.image_score, 4),
            "flags": [
                {"code": f.code, "message": f.message, "severity": f.severity}
                for f in self.risk_flags
            ],
            "return_id": self.return_id,
            "image_id": self.image_id,
            "similarity_details": self.similarity_details or {},
            "manipulation_details": self.manipulation_details or {},
        }
