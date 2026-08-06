"""
metrics.py

Computes the raw graph metrics that feed into the graph_score:

  1. weighted_degree      -- how strongly is this customer connected overall?
  2. component_size_factor -- how large is the cluster this customer sits in?
  3. community_density     -- how tightly-knit is this customer's Louvain
                               community (the actual "ring" detector)?

Each function returns a dict[customer_id -> raw_value]. Normalization to
0-1 happens later in scoring.py, kept separate so metrics stay easy to
unit test and inspect independently.
"""

import networkx as nx

try:
    import community as community_louvain  # python-louvain package
except ImportError:
    community_louvain = None


def weighted_degree(G):
    """Sum of edge weights per customer. High = strongly linked to others."""
    return {n: G.degree(n, weight="weight") for n in G.nodes()}


def component_size_factor(G):
    """
    Size of the connected component each customer belongs to.
    Isolated customers (no shared attributes with anyone) get size 1,
    which correctly implies zero graph risk from this metric.
    """
    result = {}
    for component in nx.connected_components(G):
        size = len(component)
        for n in component:
            result[n] = size
    # customers with no edges at all won't appear in G if G only has
    # nodes with connections -- callers should default missing nodes to 1.
    return result


def community_density(G):
    """
    Runs Louvain community detection, then computes each customer's
    community's internal edge density (edges_within / possible_edges).
    A dense small community = classic fraud ring signature: everyone
    connected to everyone else, not just a long chain.
    """
    if community_louvain is None:
        raise ImportError(
            "python-louvain not installed. Run: pip install python-louvain --break-system-packages"
        )

    if G.number_of_edges() == 0:
        return {n: 0.0 for n in G.nodes()}

    partition = community_louvain.best_partition(G, weight="weight")

    # group nodes by community id
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    density_by_customer = {}
    for comm_id, members in communities.items():
        subgraph = G.subgraph(members)
        n = subgraph.number_of_nodes()
        if n <= 1:
            density = 0.0
        else:
            possible_edges = n * (n - 1) / 2
            density = subgraph.number_of_edges() / possible_edges

        for node in members:
            density_by_customer[node] = density

    return density_by_customer, partition


if __name__ == "__main__":
    from data_generator import generate_dataset
    from graph_builder import build_bipartite_graph, project_customer_graph

    customers, ground_truth = generate_dataset(n_legit_customers=50, n_rings=3)
    B = build_bipartite_graph(customers)
    G = project_customer_graph(B)

    wd = weighted_degree(G)
    cs = component_size_factor(G)
    cd, partition = community_density(G)

    # show top 5 by weighted degree to eyeball whether ring members surface
    top5 = sorted(wd.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 by weighted degree:")
    for cust_id, deg in top5:
        is_ring = ground_truth.get(cust_id) is not None
        print(f"  {cust_id}: degree={deg:.1f} component_size={cs.get(cust_id,1)} "
              f"density={cd.get(cust_id, 0):.2f} is_ring_member={is_ring}")
