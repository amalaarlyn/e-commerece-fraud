"""
combined_visualization.py

Renders ONE dashboard-style figure per return request, combining:
  - The fraud-ring subgraph (left) -- from Graph Intelligence
  - Risk flags from both modules + the final decision (right panel)

This is a debug/demo visualization (matplotlib), analogous to
graph_intelligence/visualization.py's debug_plot(). The PRODUCTION
version of this is the JSON payload from combined_result.py, which
Member 1 renders interactively in React using the raw nodes/edges +
flags + decision fields -- this script is for your own presentation
and report screenshots, not what ships to the frontend.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


DECISION_COLORS = {
    "APPROVE": "#2ecc71",
    "MANUAL_REVIEW": "#f39c12",
    "REJECT": "#e74c3c",
}


def render_combined_profile(profile_dict, out_path="combined_profile.png"):
    """
    profile_dict: output of ReturnRiskProfile.to_dict()
    """
    fig, (ax_graph, ax_panel) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    # ---------------- Left: subgraph ----------------
    graph_result = profile_dict["graph_result"]
    G = nx.Graph()
    for node in graph_result["nodes"]:
        G.add_node(node["id"], risk=node.get("risk", 0.0))
    for edge in graph_result["edges"]:
        G.add_edge(edge["source"], edge["target"], relation=edge["relation"])

    focus_customer = profile_dict["customer_id"]
    pos = nx.spring_layout(G, seed=7, k=0.7)

    node_colors = [
        "#e74c3c" if n == focus_customer else "#3498db" for n in G.nodes()
    ]
    node_sizes = [900 if n == focus_customer else 500 for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color="gray", width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(
        G, pos, ax=ax_graph, node_color=node_colors, node_size=node_sizes, alpha=0.9
    )
    labels = {n: n.replace("cust_", "") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax_graph, font_size=7)

    ax_graph.set_title(
        f"Return #{profile_dict['return_id']} -- Fraud Ring Subgraph\n"
        f"(red = customer under review)",
        fontsize=11,
    )
    ax_graph.axis("off")

    # ---------------- Right: text panel ----------------
    ax_panel.axis("off")

    decision = profile_dict["decision"]["decision"]
    decision_color = DECISION_COLORS.get(decision, "black")

    y = 1.0
    ax_panel.text(
        0, y, f"DECISION: {decision}", fontsize=16, fontweight="bold",
        color=decision_color, transform=ax_panel.transAxes,
    )
    y -= 0.06
    ax_panel.text(
        0, y, profile_dict["decision"]["reasoning"], fontsize=8.5, wrap=True,
        transform=ax_panel.transAxes, va="top",
    )

    y -= 0.10
    ax_panel.text(
        0, y,
        f"Final Score: {profile_dict['final_score']:.3f}   "
        f"(graph={graph_result['graph_score']:.2f}, "
        f"image={profile_dict['image_result']['image_score']:.2f}, "
        f"behavior={profile_dict['behavior_score']:.2f})",
        fontsize=9, fontweight="bold", transform=ax_panel.transAxes,
    )

    y -= 0.08
    ax_panel.text(0, y, "Graph Intelligence Flags:", fontsize=10,
                  fontweight="bold", transform=ax_panel.transAxes)
    y -= 0.05
    for flag in graph_result["flags"]:
        ax_panel.text(0.02, y, f"• {flag['message']}", fontsize=8.5,
                      transform=ax_panel.transAxes)
        y -= 0.05

    y -= 0.03
    ax_panel.text(0, y, "Image Intelligence Flags:", fontsize=10,
                  fontweight="bold", transform=ax_panel.transAxes)
    y -= 0.05
    for flag in profile_dict["image_result"]["flags"]:
        ax_panel.text(0.02, y, f"• {flag['message']}", fontsize=8.5,
                      transform=ax_panel.transAxes)
        y -= 0.05

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    import json
    import os
    import sys

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    IMG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "image_intelligence"))

    sys.path.insert(0, SCRIPT_DIR)
    sys.path.insert(0, IMG_DIR)
    sys.path.insert(0, ROOT_DIR)

    from combined_result import build_return_risk_profile
    from data_generator import generate_dataset
    from graph_builder import build_bipartite_graph, project_customer_graph
    from scoring import compute_graph_scores
    from visualization import build_result

    # Generate sample profile dynamically
    customers, ground_truth = generate_dataset(n_legit_customers=50, n_rings=2, seed=42)
    B = build_bipartite_graph(customers)
    G = project_customer_graph(B)
    scores, explanations, partition = compute_graph_scores(G)

    top_customer = max(scores.items(), key=lambda x: x[1])[0]
    graph_res = build_result(top_customer, G, scores, explanations, partition).to_dict()

    image_res = {
        "customer_id": top_customer,
        "image_score": 0.85,
        "flags": [{"code": "NEAR_DUPLICATE_IMAGE", "message": "High similarity (85.0%) detected", "severity": 0.85}],
        "return_id": "RET_8812",
        "image_id": "IMG_8812.JPG"
    }

    profile_dict = build_return_risk_profile(
        return_id="RET_8812",
        customer_id=top_customer,
        graph_result_dict=graph_res,
        image_result_dict=image_res,
        behavior_score=0.40,
    ).to_dict()

    out_path = os.path.join(SCRIPT_DIR, "combined_dashboard.png")
    render_combined_profile(profile_dict, out_path=out_path)
    print(f"Combined dashboard visualization saved to: {out_path}")
