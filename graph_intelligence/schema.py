"""
schema.py

Central place for:
  - Node type definitions (bipartite graph: Customer <-> Attribute nodes)
  - Attribute weights (how suspicious is it to share this attribute?)
  - Typed output contracts (what Graph Intelligence hands back to
    Member 3's Risk Fusion Engine and to the frontend via the backend API)

Keeping this separate from the algorithm code means:
  1. Weights are tunable in one place without touching graph logic.
  2. When we upgrade to a GNN later, this file becomes the feature/edge-type
     schema the GNN consumes -- the contract doesn't change, only the
     scoring implementation behind it.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Node types in the bipartite graph
# ---------------------------------------------------------------------------
NODE_TYPE_CUSTOMER = "customer"
NODE_TYPE_DEVICE = "device"
NODE_TYPE_ADDRESS = "address"
NODE_TYPE_PHONE = "phone"
NODE_TYPE_PAYMENT = "payment"
NODE_TYPE_IP = "ip"

ATTRIBUTE_NODE_TYPES = [
    NODE_TYPE_DEVICE,
    NODE_TYPE_ADDRESS,
    NODE_TYPE_PHONE,
    NODE_TYPE_PAYMENT,
    NODE_TYPE_IP,
]

# ---------------------------------------------------------------------------
# Attribute weights -- how suspicious is it to SHARE this attribute
# between two "different" customers?
# Higher weight = harder to fake legitimately = stronger fraud signal.
# These are starting values for the demo; tune once real score
# distributions are visible during evaluation.
# ---------------------------------------------------------------------------
ATTRIBUTE_WEIGHTS = {
    NODE_TYPE_PAYMENT: 3.0,
    NODE_TYPE_DEVICE: 2.5,
    NODE_TYPE_ADDRESS: 2.0,
    NODE_TYPE_PHONE: 1.5,
    NODE_TYPE_IP: 1.0,
}

# ---------------------------------------------------------------------------
# Scoring weights -- how much each graph metric contributes to the
# final graph_score. Kept separate from attribute weights above.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "weighted_degree": 0.4,
    "component_size": 0.3,
    "community_density": 0.3,
}


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------
@dataclass
class RiskFlag:
    code: str          # machine-readable, e.g. "SHARED_DEVICE"
    message: str        # human-readable, e.g. "Shares device with 3 other accounts"
    severity: float      # 0-1, how much this flag alone contributes to suspicion


@dataclass
class GraphNode:
    id: str
    type: str
    risk: Optional[float] = None  # only customers carry a risk score


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str       # which attribute type is shared
    weight: float


@dataclass
class GraphIntelligenceResult:
    customer_id: str
    graph_score: float                  # 0-1, final normalized score
    risk_flags: list                    # list[RiskFlag]
    nodes: list                         # list[GraphNode]  (subgraph for visualization)
    edges: list                         # list[GraphEdge]  (subgraph for visualization)

    def to_dict(self):
        """Serializes to the exact JSON contract handed to Member 3 / frontend."""
        return {
            "customer_id": self.customer_id,
            "graph_score": round(self.graph_score, 4),
            "flags": [
                {"code": f.code, "message": f.message, "severity": f.severity}
                for f in self.risk_flags
            ],
            "nodes": [
                {"id": n.id, "type": n.type, "risk": n.risk} for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }
