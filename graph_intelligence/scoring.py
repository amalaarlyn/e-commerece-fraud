"""
scoring.py

Combines weighted_degree, component_size_factor, and community_density
into a single normalized graph_score (0-1), and generates human-readable
risk_flags for explainability (consumed by Member 3 / dashboard).

This is deliberately the "demo" implementation: a transparent weighted
formula. The future upgrade path (noted inline) replaces this function's
internals with a trained model (e.g. a GNN node-classification head)
while keeping the exact same input/output contract, so nothing else in
the pipeline needs to change.
"""

from schema import SCORE_WEIGHTS, RiskFlag, ATTRIBUTE_WEIGHTS
from metrics import weighted_degree, component_size_factor, community_density


def _normalize(value, max_value):
    """Simple min-max normalization capped at 1.0."""
    if max_value <= 0:
        return 0.0
    return min(value / max_value, 1.0)


def compute_graph_scores(G, degree_cap=15.0, component_cap=10.0):
    """
    G: projected weighted customer-customer graph

    Returns:
        scores: dict[customer_id -> graph_score]
        explanations: dict[customer_id -> list[RiskFlag]]
        partition: dict[customer_id -> community_id]  (needed later for
                   subgraph extraction / visualization)

    degree_cap / component_cap are normalization ceilings -- tune these
    once real data distributions are observed. They exist so a single
    outlier doesn't compress everyone else's score toward zero.
    """
    wd = weighted_degree(G)
    cs = component_size_factor(G)
    cd_result = community_density(G)
    density, partition = cd_result if isinstance(cd_result, tuple) else ({}, {})

    scores = {}
    explanations = {}

    for node in G.nodes():
        raw_degree = wd.get(node, 0.0)
        raw_component_size = cs.get(node, 1)
        raw_density = density.get(node, 0.0)

        norm_degree = _normalize(raw_degree, degree_cap)
        norm_component = _normalize(raw_component_size, component_cap)
        norm_density = raw_density  # already 0-1

        score = (
            SCORE_WEIGHTS["weighted_degree"] * norm_degree
            + SCORE_WEIGHTS["component_size"] * norm_component
            + SCORE_WEIGHTS["community_density"] * norm_density
        )
        scores[node] = round(score, 4)

        # --- Build explainable risk flags ---
        flags = []

        if norm_degree > 0.3:
            # figure out which attribute types drove this customer's connections
            shared_types = set()
            for neighbor in G.neighbors(node):
                shared_types.update(G[node][neighbor].get("shared_attributes", []))
            if shared_types:
                readable = ", ".join(sorted(shared_types))
                flags.append(
                    RiskFlag(
                        code="SHARED_ATTRIBUTES",
                        message=f"Shares {readable} with {G.degree(node)} other account(s)",
                        severity=round(norm_degree, 2),
                    )
                )

        if raw_component_size >= 3:
            flags.append(
                RiskFlag(
                    code="LARGE_CLUSTER",
                    message=f"Belongs to a connected cluster of {raw_component_size} accounts",
                    severity=round(norm_component, 2),
                )
            )

        if raw_density >= 0.6:
            flags.append(
                RiskFlag(
                    code="DENSE_COMMUNITY",
                    message="Part of a tightly-knit group where most accounts are interlinked "
                            "(possible organized fraud ring)",
                    severity=round(raw_density, 2),
                )
            )

        explanations[node] = flags

    return scores, explanations, partition


if __name__ == "__main__":
    from data_generator import generate_dataset
    from graph_builder import build_bipartite_graph, project_customer_graph

    customers, ground_truth = generate_dataset(n_legit_customers=50, n_rings=3)
    B = build_bipartite_graph(customers)
    G = project_customer_graph(B)

    scores, explanations, partition = compute_graph_scores(G)

    # Evaluate: are top-scored customers actually ring members?
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("Top 8 customers by graph_score:")
    for cust_id, score in ranked[:8]:
        is_ring = ground_truth.get(cust_id) is not None
        flag_msgs = [f.message for f in explanations[cust_id]]
        print(f"  {cust_id}: score={score:.3f} is_ring_member={is_ring}")
        for msg in flag_msgs:
            print(f"      - {msg}")
