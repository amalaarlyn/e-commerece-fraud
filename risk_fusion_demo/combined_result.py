"""
combined_result.py

Fuses Graph Intelligence + Image Intelligence outputs (+ a placeholder for
Member 3's behavior_score) into ONE "return risk profile" -- the unified
payload that would be persisted and shown on the admin dashboard.

This mimics what Member 3's Risk Fusion Engine will eventually do, so you
can test your two modules together end-to-end before final integration.
The fusion weights here are a stand-in; Member 3 owns the real formula.
"""

from dataclasses import dataclass, field
from decision_engine import decide, DEFAULT_COSTS


# Placeholder fusion weights -- Member 3 will own the real ones.
# Kept explicit and separate so it's obvious this is a stand-in.
DEMO_FUSION_WEIGHTS = {
    "graph_score": 0.35,
    "image_score": 0.40,
    "behavior_score": 0.25,
}


@dataclass
class ReturnRiskProfile:
    return_id: str
    customer_id: str
    graph_result: dict     # GraphIntelligenceResult.to_dict()
    image_result: dict     # ImageIntelligenceResult.to_dict()
    behavior_score: float  # placeholder, Member 3's real output
    final_score: float
    decision_result: dict  # DecisionResult.to_dict()

    def to_dict(self):
        return {
            "return_id": self.return_id,
            "customer_id": self.customer_id,
            "graph_result": self.graph_result,
            "image_result": self.image_result,
            "behavior_score": round(self.behavior_score, 4),
            "final_score": round(self.final_score, 4),
            "decision": self.decision_result,
        }


def fuse(graph_score, image_score, behavior_score, weights=None):
    w = weights or DEMO_FUSION_WEIGHTS
    return (
        w["graph_score"] * graph_score
        + w["image_score"] * image_score
        + w["behavior_score"] * behavior_score
    )


def build_return_risk_profile(
    return_id, customer_id, graph_result_dict, image_result_dict,
    behavior_score=0.3, fusion_weights=None, costs=None,
):
    """
    graph_result_dict / image_result_dict: outputs of
        GraphIntelligenceResult.to_dict() / ImageIntelligenceResult.to_dict()
    behavior_score: placeholder value standing in for Member 3's real score
    """
    graph_score = graph_result_dict.get("graph_score", 0.0)
    image_score = image_result_dict.get("image_score", 0.0)

    final_score = fuse(graph_score, image_score, behavior_score, fusion_weights)
    decision_result = decide(final_score, costs)

    return ReturnRiskProfile(
        return_id=return_id,
        customer_id=customer_id,
        graph_result=graph_result_dict,
        image_result=image_result_dict,
        behavior_score=behavior_score,
        final_score=final_score,
        decision_result=decision_result.to_dict(),
    )


if __name__ == "__main__":
    import json
    import os
    import sys

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    IMG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "image_intelligence"))

    sys.path.insert(0, IMG_DIR)
    sys.path.insert(0, ROOT_DIR)

    from data_generator import generate_dataset
    from graph_builder import build_bipartite_graph, project_customer_graph
    from scoring import compute_graph_scores
    from visualization import build_result

    # Reuse the Graph Intelligence pipeline to get a real graph_result
    customers, ground_truth = generate_dataset(n_legit_customers=200, n_rings=5, seed=42)
    B = build_bipartite_graph(customers)
    G = project_customer_graph(B)
    scores, explanations, partition = compute_graph_scores(G)

    top_customer = max(scores.items(), key=lambda x: x[1])[0]
    graph_result = build_result(top_customer, G, scores, explanations, partition).to_dict()

    # Mock image_result matching the agreed ImageIntelligenceResult.to_dict()
    # shape, standing in for your teammate's real module output (from
    # the Antigravity-built image_intelligence module).
    mock_image_result = {
        "customer_id": top_customer,
        "image_score": 0.95,
        "return_id": "RET_CLAIM_9944",
        "image_id": "IMG_9944_TAMPERED.JPG",
        "flags": [
            {
                "code": "EXACT_DUPLICATE_IMAGE",
                "message": "Identical image previously submitted with a different return claim",
                "severity": 0.98,
            }
        ],
    }

    profile = build_return_risk_profile(
        return_id="RET_CLAIM_9944",
        customer_id=top_customer,
        graph_result_dict=graph_result,
        image_result_dict=mock_image_result,
        behavior_score=0.4,
    )

    print("=" * 70)
    print("COMBINED RETURN RISK PROFILE")
    print("=" * 70)
    print(json.dumps(profile.to_dict(), indent=2))

    with open("sample_combined_profile.json", "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    print("\nSaved to sample_combined_profile.json")
