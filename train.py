"""
train.py
--------
Trains two versions of the LSTM fraud model:
  1. BASELINE   - standard BCE loss (treats FP and FN equally)
  2. COST-AWARE - asymmetric loss where a False Positive (blocking a genuine
                  customer) is weighted MORE heavily than a False Negative
                  (missing one fraud case) -- reflecting the real business cost
                  asymmetry described in the problem statement.

Also implements:
  - Probability calibration (Platt scaling) so scores are meaningful, not just rankings
  - A three-tier decision system (auto-approve / soft-friction / deny) instead of
    a single harsh binary cutoff -- this is the main false-positive mitigation lever.
  - A simple $-cost comparison between baseline and cost-aware models.
  - A Random Forest on aggregated tabular features as the "prior work" baseline
    (mirrors Ravala & Asadinia 2025's approach) for reference.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ReturnFraudLSTM

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "/home/claude/return_fraud/data"
OUT_DIR = "/home/claude/return_fraud/outputs"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# --- Business cost assumptions (tune these with real business numbers later) ---
# Stated in the problem: cost of alienating ONE genuine customer > avg value of ONE fraud case.
COST_FALSE_POSITIVE = 150.0   # est. cost of wrongly blocking a genuine customer (support, churn, escalation)
COST_FALSE_NEGATIVE = 70.0    # est. avg $ lost per missed fraudulent return
# Loss weighting ratio directly reflects this asymmetry.
FP_WEIGHT = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)  # weight on negative-class errors
FN_WEIGHT = COST_FALSE_NEGATIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)  # weight on positive-class errors


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def asymmetric_bce_loss(logits, targets, fp_weight, fn_weight):
    """
    Custom weighted BCE:
      - positive samples (actual fraud) weighted by fn_weight (controls miss cost)
      - negative samples (genuine) weighted by fp_weight (controls false-alarm cost)
    Higher fp_weight -> model becomes MORE cautious about flagging fraud
    (fewer false positives, at the cost of possibly more false negatives).
    """
    probs = torch.sigmoid(logits)
    eps = 1e-7
    probs = torch.clamp(probs, eps, 1 - eps)
    loss = -(fn_weight * targets * torch.log(probs) + fp_weight * (1 - targets) * torch.log(1 - probs))
    return loss.mean()


def train_model(X_train, y_train, X_val, y_val, cost_aware=False, epochs=25, lr=1e-3, batch_size=64):
    model = ReturnFraudLSTM(n_features=X_train.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            if cost_aware:
                loss = asymmetric_bce_loss(logits, yb, FP_WEIGHT, FN_WEIGHT)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val, dtype=torch.float32))
                val_probs = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(y_val, val_probs)
            print(f"  [{'cost-aware' if cost_aware else 'baseline'}] epoch {epoch+1}/{epochs} "
                  f"train_loss={total_loss/len(train_loader.dataset):.4f} val_auc={val_auc:.4f}")
    return model


def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def calibrate(raw_probs_val, y_val, raw_probs_test):
    """Platt scaling: fit a 1D logistic regression on raw scores -> calibrated probability.
    Returns (calibrated_test_probs, fitted_calibrator) -- the calibrator is returned so it
    can be persisted and reused at inference time (see risk_api.py)."""
    lr = LogisticRegression()
    lr.fit(raw_probs_val.reshape(-1, 1), y_val)
    return lr.predict_proba(raw_probs_test.reshape(-1, 1))[:, 1], lr


def three_tier_decision(probs, low_thresh=0.30, high_thresh=0.70):
    """
    Returns tier labels: 'auto_approve', 'soft_friction', 'deny'
    instead of a single harsh cutoff -- this directly reduces false-positive HARM
    even when the model isn't perfectly certain.
    """
    tiers = np.where(probs < low_thresh, "auto_approve",
             np.where(probs < high_thresh, "soft_friction", "deny"))
    return tiers


def evaluate(y_true, probs, label, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)

    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    tp = np.sum((preds == 1) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    total_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE

    print(f"\n--- {label} (threshold={threshold}) ---")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  AUC: {auc:.3f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Estimated business cost: ${total_cost:,.0f}  (FP cost=${COST_FALSE_POSITIVE}, FN cost=${COST_FALSE_NEGATIVE})")
    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc,
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn), "cost": float(total_cost)}


def fpr_at_fixed_recall(y_true, probs, target_recall=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    idx = np.argmin(np.abs(tpr - target_recall))
    return fpr[idx], thresholds[idx]


def rf_tabular_baseline(meta_train, meta_test, y_train, y_test):
    """Mirrors prior work (Ravala & Asadinia 2025): tabular-only Random Forest."""
    def encode(meta):
        cat_dummies = pd.get_dummies(meta["category"], prefix="cat")
        reason_dummies = pd.get_dummies(meta["return_reason"], prefix="reason")
        feats = pd.concat([meta[["order_value"]].reset_index(drop=True),
                            cat_dummies.reset_index(drop=True),
                            reason_dummies.reset_index(drop=True)], axis=1)
        return feats

    Xtr = encode(meta_train)
    Xte = encode(meta_test)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(Xtr, y_train)
    probs = rf.predict_proba(Xte)[:, 1]
    return probs


def main():
    X = np.load(f"{DATA_DIR}/X_hybrid.npy")
    y = np.load(f"{DATA_DIR}/y_hybrid.npy")
    meta = pd.read_csv(f"{DATA_DIR}/meta_hybrid.csv")

    # Split by CUSTOMER to avoid leakage (same customer's history shouldn't span train/test)
    customer_ids = meta["customer_id"].unique()
    train_ids, temp_ids = train_test_split(customer_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_mask = meta["customer_id"].isin(train_ids).values
    val_mask = meta["customer_id"].isin(val_ids).values
    test_mask = meta["customer_id"].isin(test_ids).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_train, meta_val, meta_test = meta[train_mask], meta[val_mask], meta[test_mask]

    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print(f"Cost weights -> FP_WEIGHT={FP_WEIGHT:.3f}, FN_WEIGHT={FN_WEIGHT:.3f}\n")

    # --- Train baseline LSTM ---
    print("Training BASELINE LSTM (standard BCE)...")
    baseline_model = train_model(X_train, y_train, X_val, y_val, cost_aware=False)
    baseline_probs_test = get_probs(baseline_model, X_test)

    # --- Train cost-aware LSTM ---
    print("\nTraining COST-AWARE LSTM (asymmetric loss)...")
    cost_model = train_model(X_train, y_train, X_val, y_val, cost_aware=True)
    cost_probs_val = get_probs(cost_model, X_val)
    cost_probs_test_raw = get_probs(cost_model, X_test)

    # --- Calibrate cost-aware model's probabilities using validation set ---
    cost_probs_test_calibrated, calibrator = calibrate(cost_probs_val, y_val, cost_probs_test_raw)

    # --- Random Forest tabular baseline (prior-work style) ---
    print("\nTraining Random Forest tabular baseline (prior-work style)...")
    rf_probs_test = rf_tabular_baseline(meta_train, meta_test, y_train, y_test)

    # --- Evaluate all three ---
    results = {}
    results["baseline_lstm"] = evaluate(y_test, baseline_probs_test, "Baseline LSTM")
    results["cost_aware_lstm"] = evaluate(y_test, cost_probs_test_calibrated, "Cost-Aware LSTM (calibrated)")
    results["rf_tabular"] = evaluate(y_test, rf_probs_test, "Random Forest (tabular, prior-work style)")

    # --- FPR at fixed recall (key false-positive metric) ---
    print("\n--- False Positive Rate at fixed 90% recall ---")
    for name, probs in [("Baseline LSTM", baseline_probs_test),
                         ("Cost-Aware LSTM", cost_probs_test_calibrated),
                         ("RF tabular", rf_probs_test)]:
        fpr90, thresh90 = fpr_at_fixed_recall(y_test, probs, 0.90)
        print(f"{name}: FPR@90%recall = {fpr90:.3f} (threshold={thresh90:.3f})")

    # --- Three-tier decision system on cost-aware model ---
    tiers = three_tier_decision(cost_probs_test_calibrated)
    tier_counts = pd.Series(tiers).value_counts()
    print("\n--- Three-tier decision breakdown (cost-aware model) ---")
    print(tier_counts)
    tier_df = pd.DataFrame({"true_label": y_test, "tier": tiers})
    print("\nFraud rate within each tier (sanity check -- deny tier should be highest):")
    print(tier_df.groupby("tier")["true_label"].mean())

    # --- ROC curve plot ---
    plt.figure(figsize=(6, 5))
    for name, probs in [("Baseline LSTM", baseline_probs_test),
                         ("Cost-Aware LSTM", cost_probs_test_calibrated),
                         ("RF tabular (prior-work style)", rf_probs_test)]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Comparison: Hybrid (Real UCI + Synthetic Fraud)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_comparison_hybrid.png", dpi=150)
    print(f"\nSaved ROC plot to {OUT_DIR}/roc_comparison_hybrid.png")

    # --- Save summary table ---
    summary = pd.DataFrame(results).T
    summary.to_csv(f"{OUT_DIR}/results_summary_hybrid.csv")
    print(f"\nSaved results summary to {OUT_DIR}/results_summary_hybrid.csv")
    print(summary)

    # --- Persist artifacts for inference (risk_api.py) ---
    # The cost-aware, calibrated model is the one that goes into production --
    # it's the one whose score is used for the three-tier decision above.
    torch.save(cost_model.state_dict(), f"{OUT_DIR}/cost_aware_lstm.pt")
    joblib.dump(calibrator, f"{OUT_DIR}/calibrator.joblib")
    with open(f"{OUT_DIR}/decision_config.json", "w") as f:
        json.dump({
            "low_thresh": 0.30,
            "high_thresh": 0.70,
            "cost_false_positive": COST_FALSE_POSITIVE,
            "cost_false_negative": COST_FALSE_NEGATIVE,
            "n_features": int(X_train.shape[-1]),
            "seq_len": int(X_train.shape[1]),
        }, f, indent=2)
    print(f"\nSaved inference artifacts (model, calibrator, thresholds) to {OUT_DIR}")


if __name__ == "__main__":
    main()
