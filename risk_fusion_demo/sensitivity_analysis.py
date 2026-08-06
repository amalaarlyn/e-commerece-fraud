"""
sensitivity_analysis.py

Sweeps cost_false_reject and cost_missed_fraud across a range of values
and records the resulting effective decision thresholds (from
decision_engine.find_effective_thresholds). This is the evidence for
your evaluation chapter that decision boundaries shift predictably and
sensibly as business costs change -- proving the cost-asymmetric design
actually does what it claims, rather than just asserting it.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decision_engine import find_effective_thresholds, DEFAULT_COSTS


def sweep_false_reject_cost(values, base_costs=None):
    base = dict(base_costs or DEFAULT_COSTS)
    results = []
    for v in values:
        costs = dict(base, cost_false_reject=v)
        bands = find_effective_thresholds(costs)
        results.append((v, bands))
    return results


def plot_sensitivity(results, out_path="sensitivity_analysis.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {"APPROVE": "#2ecc71", "MANUAL_REVIEW": "#f39c12", "REJECT": "#e74c3c"}

    for i, (cost_val, bands) in enumerate(results):
        for decision, start, end in bands:
            ax.barh(
                y=i, width=(end - start), left=start, height=0.6,
                color=color_map[decision], edgecolor="white",
            )

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"cost_false_reject = {v:.0f}" for v, _ in results])
    ax.set_xlabel("Final fraud score")
    ax.set_xlim(0, 1)
    ax.set_title(
        "How decision thresholds shift as cost_false_reject changes\n"
        "(cost_missed_fraud and cost_manual_review held constant)"
    )

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys(), loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    values = [20, 40, 60, 80, 100, 150, 200]
    results = sweep_false_reject_cost(values)

    print("=" * 70)
    print("SENSITIVITY ANALYSIS -- effective decision bands vs cost_false_reject")
    print("=" * 70)
    for cost_val, bands in results:
        band_str = "  ".join(f"{d}[{s:.2f}-{e:.2f}]" for d, s, e in bands)
        print(f"cost_false_reject={cost_val:>5.0f}:  {band_str}")

    out_path = os.path.join(SCRIPT_DIR, "sensitivity_analysis.png")
    plot_sensitivity(results, out_path)
    print(f"\nSaved plot to: {out_path}")
