"""
graph_builder.py

Builds the bipartite graph (Customer <-> Attribute nodes) from customer
records, then projects it into a weighted Customer-Customer graph.

Why bipartite first, then project:
  - The bipartite graph is the richer, "ground truth" structure -- it's
    exactly what a future GNN (e.g. PyTorch Geometric heterogeneous graph)
    would consume directly, so building it now means the upgrade later
    doesn't require re-deriving relationships from scratch.
  - The projected Customer-Customer graph is what today's demo algorithms
    (degree, connected components, Louvain) and the visualization actually
    operate on, since those are customer-centric questions.
"""

import networkx as nx
from itertools import combinations
from collections import defaultdict

from schema import ATTRIBUTE_WEIGHTS, NODE_TYPE_CUSTOMER


ATTRIBUTE_FIELD_TO_TYPE = {
    "device": "device",
    "address": "address",
    "phone": "phone",
    "payment": "payment",
    "ip": "ip",
}


def build_bipartite_graph(customers):
    """
    customers: list[CustomerRecord]

    Returns a networkx.Graph with:
      - customer nodes:  type='customer'
      - attribute nodes: type=<device|address|phone|payment|ip>
      - edges: customer -- attribute_value, relation=<attribute type>
    """
    B = nx.Graph()

    for c in customers:
        B.add_node(c.customer_id, type=NODE_TYPE_CUSTOMER)

        for field_name, attr_type in ATTRIBUTE_FIELD_TO_TYPE.items():
            attr_value = getattr(c, field_name)
            # attribute node id is the value itself (already globally unique
            # per generator, e.g. "device_2c62fa3e") -- this is what makes
            # two customers "connected" if they share the same value.
            if not B.has_node(attr_value):
                B.add_node(attr_value, type=attr_type)
            B.add_edge(c.customer_id, attr_value, relation=attr_type)

    return B


def project_customer_graph(bipartite_graph):
    """
    Projects the bipartite graph into a weighted Customer-Customer graph.

    Two customers get an edge if they share at least one attribute node.
    Edge weight = sum of ATTRIBUTE_WEIGHTS for every attribute type they
    share (a pair sharing both device AND payment gets a heavier edge
    than a pair sharing only IP).

    Also stores which attribute types were shared, for explainability.
    """
    G = nx.Graph()

    customer_nodes = [
        n for n, d in bipartite_graph.nodes(data=True) if d["type"] == NODE_TYPE_CUSTOMER
    ]
    G.add_nodes_from(customer_nodes)

    # For each attribute node, all customers connected to it are "linked"
    # via that attribute. Group customers by shared attribute node.
    edge_shared_types = defaultdict(set)   # (cust_a, cust_b) -> {"device", "ip", ...}
    edge_weight = defaultdict(float)

    for node, data in bipartite_graph.nodes(data=True):
        if data["type"] == NODE_TYPE_CUSTOMER:
            continue  # only process attribute nodes here

        attr_type = data["type"]
        linked_customers = list(bipartite_graph.neighbors(node))

        if len(linked_customers) < 2:
            continue  # attribute used by only one customer -> no signal

        weight = ATTRIBUTE_WEIGHTS.get(attr_type, 1.0)

        for cust_a, cust_b in combinations(sorted(linked_customers), 2):
            key = (cust_a, cust_b)
            edge_shared_types[key].add(attr_type)
            edge_weight[key] += weight

    for (cust_a, cust_b), weight in edge_weight.items():
        G.add_edge(
            cust_a,
            cust_b,
            weight=weight,
            shared_attributes=sorted(edge_shared_types[(cust_a, cust_b)]),
        )

    return G


if __name__ == "__main__":
    from data_generator import generate_dataset

    customers, ground_truth = generate_dataset(n_legit_customers=50, n_rings=3)
    B = build_bipartite_graph(customers)
    G = project_customer_graph(B)

    print(f"Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"Projected customer graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # show a sample edge to confirm shared_attributes + weight look right
    sample_edge = list(G.edges(data=True))[0]
    print("Sample projected edge:", sample_edge)
