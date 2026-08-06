"""
explainability.py
------------------
Turns a raw risk score from ReturnFraudLSTM into a human-readable explanation.

Approach: Integrated Gradients (Sundararajan et al., 2017), implemented from
scratch (no captum dependency) -- it attributes the model's output back to
each (timestep, feature) cell in the input sequence by integrating gradients
along a straight-line path from a zero baseline to the actual input.

This gives two things per prediction:
  1. Which FEATURE mattered most (e.g. "days_to_return" vs "order_value")
  2. Which PAST ORDER (timestep) in the sequence mattered most

That maps directly onto the "Explainability Layer" box in the behavior
analysis pipeline, and downstream onto Module 5 (LLM Fraud Investigator) --
this rule-based version is what actually runs on every request (cheap,
deterministic); an LLM can later be layered on top to turn these structured
attributions into prose, using this output as its grounding/RAG context
instead of hallucinating a reason.
"""

import numpy as np
import torch

from features import SEQ_LEN, N_FEATURES, REASON_VOCAB

FEATURE_NAMES = [
    "recency",              # days since previous order (normalized)
    "order_value",          # log order value (normalized)
    "high_value_category",  # apparel_event / electronics flag
    "returned_flag",        # was this order returned
    "days_to_return",       # speed of return (normalized)
    "delivery_confirmed",   # carrier delivery status
    "return_reason",        # encoded reason ordinal
    "rolling_return_rate",          # returns / orders so far (before this order)
    "rolling_refund_to_spend",      # $ refunded / $ spent so far (before this order)
    "category_concentration",       # share of past returns in this customer's top category
    "days_since_last_return",       # recency of returns specifically, not just orders
]


def integrated_gradients(model, x, steps=50):
    """
    x: torch.Tensor, shape (seq_len, n_features) -- a single sample, no batch dim.
    Returns: torch.Tensor, same shape, attribution per (timestep, feature) cell.
    """
    model.eval()
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps + 1)

    grads = []
    for alpha in alphas:
        interpolated = (baseline + alpha * (x - baseline)).clone().requires_grad_(True)
        logit = model(interpolated.unsqueeze(0)).squeeze()
        model.zero_grad()
        logit.backward()
        grads.append(interpolated.grad.detach().clone())

    avg_grad = torch.stack(grads).mean(dim=0)
    attributions = (x - baseline) * avg_grad
    return attributions  # (seq_len, n_features)


def _reason_from_encoding(encoded_value):
    """Reverses features.py's `reason_onehot.index(1) / len(REASON_VOCAB)` encoding."""
    idx = int(round(encoded_value * len(REASON_VOCAB)))
    idx = max(0, min(idx, len(REASON_VOCAB) - 1))
    return REASON_VOCAB[idx]


def _feature_phrase(feature_name, cell_value, attribution_sign):
    """Rule-based natural-language phrase for one (feature, value, direction) triple."""
    risky = attribution_sign > 0  # positive attribution = pushed score toward fraud

    if feature_name == "days_to_return" and risky and cell_value < 0.1:
        return "returned unusually fast after purchase"
    if feature_name == "high_value_category" and risky and cell_value > 0.5:
        return "return involves a high-value / high-risk category (event wear or electronics)"
    if feature_name == "recency" and risky and cell_value < 0.05:
        return "ordered again almost immediately after the previous order"
    if feature_name == "order_value" and risky and cell_value > 0.6:
        return "unusually high order value relative to this customer's history"
    if feature_name == "delivery_confirmed" and risky and cell_value < 0.5:
        return "delivery was not confirmed by the carrier"
    if feature_name == "return_reason":
        reason = _reason_from_encoding(cell_value)
        if risky and reason == "item_not_received":
            return "reason given was 'item not received', a common INR-abuse pattern"
        if risky and reason == "changed_mind":
            return "reason given was 'changed mind', consistent with wardrobing"
        if risky and reason == "damaged":
            return "reason given was 'damaged' -- flag for image forensics cross-check"
    if feature_name == "rolling_return_rate" and risky and cell_value > 0.4:
        return "this customer already returns a high share of what they order"
    if feature_name == "rolling_refund_to_spend" and risky and cell_value > 0.4:
        return "a large fraction of this customer's total spend has come back as refunds"
    if feature_name == "category_concentration" and risky and cell_value > 0.6:
        return "this customer's past returns are heavily concentrated in one category"
    if feature_name == "days_since_last_return" and risky and cell_value < 0.15:
        return "returned again very soon after their last return -- unusual rhythm"
    return f"{feature_name} contributed {'toward' if risky else 'away from'} the fraud signal"


def explain_prediction(model, x_sample, top_k=3):
    """
    x_sample: np.array or torch.Tensor, shape (SEQ_LEN, N_FEATURES) -- the exact
    window fed to the model for one return-event prediction.

    Returns a dict:
      - top_features: list of (feature_name, importance) sorted descending
      - top_timesteps: list of (steps_ago, importance) sorted descending
      - reasons: list of human-readable phrases, most important first
    """
    x = torch.as_tensor(x_sample, dtype=torch.float32)
    attributions = integrated_gradients(model, x)  # (seq_len, n_features)

    feature_importance = attributions.abs().sum(dim=0)          # (n_features,)
    timestep_importance = attributions.abs().sum(dim=1)         # (seq_len,)

    top_feat_idx = torch.argsort(feature_importance, descending=True)[:top_k]
    top_step_idx = torch.argsort(timestep_importance, descending=True)[:top_k]

    top_features = [(FEATURE_NAMES[i], round(feature_importance[i].item(), 4)) for i in top_feat_idx]
    seq_len = x.shape[0]
    top_timesteps = [(seq_len - i.item(), round(timestep_importance[i].item(), 4)) for i in top_step_idx]

    reasons = []
    for i in top_feat_idx:
        i = i.item()
        # look at the most recent (last) timestep's cell for that feature -- the
        # return event being scored is always the final row in the window
        cell_value = x[-1, i].item()
        cell_attr = attributions[-1, i].item()
        reasons.append(_feature_phrase(FEATURE_NAMES[i], cell_value, cell_attr))

    return {
        "top_features": top_features,
        "top_timesteps": top_timesteps,  # "N orders ago" -> importance
        "reasons": reasons,
    }
