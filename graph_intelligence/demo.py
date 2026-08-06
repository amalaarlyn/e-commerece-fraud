"""
demo.py

Runs the full Graph Intelligence pipeline end-to-end:

  1. Generate synthetic customers (with injected fraud rings + ground truth)
  2. Build bipartite graph -> project to weighted customer-customer graph
  3. Compute graph_score + risk_flags for every customer
  4. Evaluate against ground truth (precision/recall at a chosen threshold)
  5. Export the top flagged customer's result in the exact JSON contract
     that Member 3 / the frontend will consume
  6. Save a debug visualization

This is the script to run and show as "Graph Intelligence v1 working demo."
"""

import json

from data_generator import generate_dataset
from graph_builder import build_bipartite_graph, project_customer_graph
from scoring import compute_graph_scores
from visualization import build_result, debug_plot


def evaluate(scores, ground_truth, all_customer_ids, threshold=0.5):
    """
    Precision / Recall / F1 of graph_score >= threshold as a fraud-ring
    predictor, against injected ground truth.

    Customers with no attribute overlap never appear in G (isolated
    nodes are not added to the projected graph in our current builder),
    so we treat any customer missing from `scores` as score 0.0.
    """
    tp = fp = fn = tn = 0

    for cust_id in all_customer_ids:
        predicted_fraud = scores.get(cust_id, 0.0) >= threshold
        actual_fraud = ground_truth.get(cust_id) is not None

        if predicted_fraud and actual_fraud:
            tp += 1
        elif predicted_fraud and not actual_fraud:
            fp += 1
        elif not predicted_fraud and actual_fraud:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def run_demo():
    print("=" * 70)
    print("GRAPH INTELLIGENCE MODULE -- END TO END DEMO")
    print("=" * 70)

    # 1. Generate data
    customers, ground_truth = generate_dataset(
        n_legit_customers=200, n_rings=5, seed=42
    )
    all_customer_ids = [c.customer_id for c in customers]
    print(f"\n[1] Generated {len(customers)} customers "
          f"({len(ground_truth)} are injected fraud-ring members)")

    # 2. Build graph
    B = build_bipartite_graph(customers)
    G = project_customer_graph(B)
    print(f"[2] Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"    Projected customer graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # 3. Score everyone
    scores, explanations, partition = compute_graph_scores(G)
    print(f"[3] Computed graph_score for {len(scores)} customers "
          f"(customers with zero attribute overlap default to score 0.0)")

    # 4. Evaluate
    metrics = evaluate(scores, ground_truth, all_customer_ids, threshold=0.5)
    print(f"\n[4] Evaluation at threshold={metrics['threshold']}:")
    print(f"    Precision: {metrics['precision']}   Recall: {metrics['recall']}   "
          f"F1: {metrics['f1']}")
    print(f"    TP={metrics['true_positives']}  FP={metrics['false_positives']}  "
          f"FN={metrics['false_negatives']}  TN={metrics['true_negatives']}")

    # 5. Export sample JSON contract for the highest-risk customer
    top_customer = max(scores.items(), key=lambda x: x[1])[0]
    result = build_result(top_customer, G, scores, explanations, partition)
    output_path = "/home/claude/graph_intelligence/sample_output.json"
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\n[5] Sample output JSON (for Member 3 / frontend) saved to:\n    {output_path}")

    # 6. Debug plot
    plot_path = debug_plot(G, ground_truth, path="/home/claude/graph_intelligence/graph_debug.png")
    print(f"[6] Debug visualization saved to:\n    {plot_path}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
