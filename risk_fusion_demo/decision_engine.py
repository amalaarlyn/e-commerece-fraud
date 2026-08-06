"""
decision_engine.py

Implements the COST-ASYMMETRIC decision logic that is the core novelty of
the project: instead of thresholding a fraud probability at a fixed cutoff
(which optimizes accuracy), this picks whichever action -- Approve, Reject,
Manual Review -- has the LOWEST EXPECTED BUSINESS COST, given the fused
fraud score.

This module is a PROTOTYPE for testing the full pipeline end-to-end on
your side. The final production version of this logic belongs to
Member 3's Risk Fusion Engine -- treat this as the reference design you
hand off, not a replacement for their module.

Design:

    E[cost | Approve] = final_score * COST_MISSED_FRAUD
    E[cost | Reject]  = (1 - final_score) * COST_FALSE_REJECT
    E[cost | Review]  = COST_MANUAL_REVIEW   (fixed cost, independent of score)

    decision = argmin over the three expected costs

Because COST_FALSE_REJECT is set much higher than COST_MISSED_FRAUD, the
system requires much stronger evidence before it ever rejects outright,
and prefers Manual Review across a wide middle band of scores -- this is
the literal mathematical expression of "false-positive cost >> fraud cost."
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Business cost configuration -- THESE ARE THE KNOBS THAT ENCODE YOUR
# PROJECT'S CORE THESIS. Tune these and show the resulting threshold shift
# in your evaluation section; it's a strong demonstration of the idea.
# ---------------------------------------------------------------------------
DEFAULT_COSTS = {
    "cost_false_reject": 100.0,   # blocking a genuine customer (lost CLV, trust)
    "cost_missed_fraud": 25.0,    # approving one fraudulent return
    "cost_manual_review": 8.0,    # human review labor cost (fixed)
}


@dataclass
class DecisionResult:
    final_score: float
    decision: str                 # "APPROVE" | "MANUAL_REVIEW" | "REJECT"
    expected_costs: dict           # {"approve": ..., "reject": ..., "review": ...}
    reasoning: str                 # human-readable explanation of why this decision won

    def to_dict(self):
        return {
            "final_score": round(self.final_score, 4),
            "decision": self.decision,
            "expected_costs": {k: round(v, 3) for k, v in self.expected_costs.items()},
            "reasoning": self.reasoning,
        }


def decide(final_score: float, costs: dict = None) -> DecisionResult:
    """
    final_score: fused fraud probability in [0, 1] (from Member 3's
                 Risk Fusion Engine, combining graph_score, image_score,
                 behavior_score)
    costs: override DEFAULT_COSTS for sensitivity analysis / experiments
    """
    c = costs or DEFAULT_COSTS

    expected_costs = {
        "approve": final_score * c["cost_missed_fraud"],
        "reject": (1 - final_score) * c["cost_false_reject"],
        "review": c["cost_manual_review"],
    }

    decision_key = min(expected_costs, key=expected_costs.get)
    decision_map = {
        "approve": "APPROVE",
        "reject": "REJECT",
        "review": "MANUAL_REVIEW",
    }
    decision = decision_map[decision_key]

    sorted_costs = sorted(expected_costs.items(), key=lambda x: x[1])
    cheapest, second = sorted_costs[0], sorted_costs[1]
    margin = second[1] - cheapest[1]

    reasoning = (
        f"At final_score={final_score:.3f}, expected cost of "
        f"{decision} (${cheapest[1]:.1f}) is lowest, "
        f"{margin:.1f} cheaper than next best option ({decision_map[second[0]]}, "
        f"${second[1]:.1f})."
    )

    return DecisionResult(
        final_score=final_score,
        decision=decision,
        expected_costs=expected_costs,
        reasoning=reasoning,
    )


def find_effective_thresholds(costs: dict = None, resolution: int = 1000):
    """
    Sweeps final_score from 0 to 1 to find the ACTUAL score bands each
    decision wins at, given the cost configuration. Useful for showing,
    in your report, what fixed-looking thresholds these dynamic costs
    imply -- e.g. "APPROVE below 0.19, MANUAL_REVIEW 0.19-0.76, REJECT
    above 0.76" -- and how that shifts if costs change.
    """
    c = costs or DEFAULT_COSTS
    bands = []
    current_decision = None
    band_start = 0.0

    for i in range(resolution + 1):
        score = i / resolution
        result = decide(score, c)
        if result.decision != current_decision:
            if current_decision is not None:
                bands.append((current_decision, band_start, score))
            current_decision = result.decision
            band_start = score

    bands.append((current_decision, band_start, 1.0))
    return bands


if __name__ == "__main__":
    print("=" * 70)
    print("DECISION ENGINE -- COST-SENSITIVE THRESHOLD DEMO")
    print("=" * 70)

    test_scores = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
    print(f"\nUsing default costs: {DEFAULT_COSTS}\n")
    for score in test_scores:
        result = decide(score)
        print(f"score={score:.2f} -> {result.decision:14s} | {result.reasoning}")

    print("\nEffective score bands (given these costs):")
    for decision, start, end in find_effective_thresholds():
        print(f"  {decision:14s}: [{start:.3f}, {end:.3f})")

    print("\n--- Sensitivity check: what if false-reject cost is LOWER (e.g. 40) ---")
    lenient_costs = dict(DEFAULT_COSTS, cost_false_reject=40.0)
    for decision, start, end in find_effective_thresholds(lenient_costs):
        print(f"  {decision:14s}: [{start:.3f}, {end:.3f})")
