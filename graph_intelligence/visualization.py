"""
visualization.py

Two responsibilities:
  1. extract_subgraph(): pulls out the relevant local neighborhood around
     a flagged customer (not the whole dataset graph) -- this is what
     actually gets rendered on the dashboard.
  2. build_result(): assembles the final GraphIntelligenceResult object
     (schema.py contract) that gets serialized to JSON for Member 3 /
     the frontend.

Also includes a debug-only matplotlib plot, for our own sanity checking
during development -- NOT part of the production output contract.
"""

import networkx as nx

from schema import GraphIntelligenceResult, GraphNode, GraphEdge, NODE_TYPE_CUSTOMER


def extract_subgraph(G, customer_id, partition=None, hops=1):
    """
    Returns the local subgraph around `customer_id` up to `hops` edges away.
    Defaults to 1 hop: the customer plus everyone they're directly
    connected to -- enough to visualize "who is this customer linked to
    and why," without dumping the entire dataset graph onto the dashboard.
    """
    if customer_id not in G:
        return nx.Graph()

    nodes_in_range = {customer_id}
    frontier = {customer_id}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.neighbors(n))
        nodes_in_range.update(next_frontier)
        frontier = next_frontier

    return G.subgraph(nodes_in_range).copy()


def build_result(customer_id, G, scores, explanations, partition=None, hops=1):
    """
    Assembles a GraphIntelligenceResult for a single customer -- this is
    the exact object whose .to_dict() gets sent to Member 3's Risk Fusion
    Engine and, from there, persisted for the frontend dashboard.
    """
    subgraph = extract_subgraph(G, customer_id, partition=partition, hops=hops)

    nodes = [
        GraphNode(
            id=n,
            type=NODE_TYPE_CUSTOMER,
            risk=scores.get(n),
        )
        for n in subgraph.nodes()
    ]

    edges = [
        GraphEdge(
            source=u,
            target=v,
            relation="+".join(data.get("shared_attributes", [])),
            weight=data.get("weight", 0.0),
        )
        for u, v, data in subgraph.edges(data=True)
    ]

    return GraphIntelligenceResult(
        customer_id=customer_id,
        graph_score=scores.get(customer_id, 0.0),
        risk_flags=explanations.get(customer_id, []),
        nodes=nodes,
        edges=edges,
    )


def debug_plot(G, ground_truth=None, path="graph_debug.png"):
    """
    Dev-only visualization to eyeball whether fraud rings visually cluster
    as expected. Uses matplotlib -- this is NOT what ships to the
    frontend; Member 1 renders the JSON contract with a React graph
    library (e.g. react-force-graph, cytoscape.js) instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(G, seed=42, k=0.5)
    colors = []
    for n in G.nodes():
        if ground_truth and ground_truth.get(n):
            colors.append("red")
        else:
            colors.append("steelblue")

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos, node_color=colors, node_size=60, with_labels=False,
        edge_color="gray", width=0.5, alpha=0.8,
    )
    plt.title("Customer graph (red = actual fraud ring member, ground truth)")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    from data_generator import generate_dataset
    from graph_builder import build_bipartite_graph, project_customer_graph
    from scoring import compute_graph_scores
    import json

    customers, ground_truth = generate_dataset(n_legit_customers=50, n_rings=3)
    B = build_bipartite_graph(customers)
    G = project_customer_graph(B)
    scores, explanations, partition = compute_graph_scores(G)

    # pick a top-scored customer to demo the output contract
    top_customer = max(scores.items(), key=lambda x: x[1])[0]
    result = build_result(top_customer, G, scores, explanations, partition)

    print("Sample output JSON for customer:", top_customer)
    print(json.dumps(result.to_dict(), indent=2))

    debug_plot(G, ground_truth, path="/home/claude/graph_intelligence/graph_debug.png")
    print("\nDebug plot saved.")
