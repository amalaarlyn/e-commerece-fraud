"""
risk_api.py
-----------
Serves the trained behavior model as a live scoring endpoint.

Request:  a customer's raw order history (list of order dicts), ending with
          the return event currently being evaluated.
Response: calibrated risk score, decision tier (auto_approve / soft_friction
          / deny), and a structured explanation.

This is the "Risk API" box at the bottom of the architecture diagram --
everything upstream (features.py, model.py, train_hybrid.py,
explainability.py) exists to produce the three artifacts this file loads:
cost_aware_lstm.pt, calibrator.joblib, decision_config.json.

Run:
    uvicorn risk_api:app --reload --port 8000

Example request body (POST /score):
{
  "customer_id": "cust_123",
  "orders": [
    {"order_date": "2026-05-01T10:00:00", "category": "electronics",
     "order_value": 899.0, "returned": 0, "return_date": null,
     "return_reason": null, "delivery_confirmed": 1},
    {"order_date": "2026-06-14T09:00:00", "category": "apparel_event",
     "order_value": 240.0, "returned": 1, "return_date": "2026-06-16T09:00:00",
     "return_reason": "changed_mind", "delivery_confirmed": 1}
  ]
}
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from features import SEQ_LEN, N_FEATURES, REASON_VOCAB, encode_reason
    from model import ReturnFraudLSTM
    from explainability import explain_prediction
except ImportError:
    SEQ_LEN = 8
    N_FEATURES = 11
    REASON_VOCAB = ["none", "wrong_size", "changed_mind", "wrong_item", "damaged", "item_not_received"]
    def encode_reason(reason):
        vec = [0] * len(REASON_VOCAB)
        idx = REASON_VOCAB.index(reason) if reason in REASON_VOCAB else 0
        vec[idx] = 1
        return vec
    class ReturnFraudLSTM(torch.nn.Module):
        def __init__(self, n_features: int = 11, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.lstm = torch.nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 16),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(16, 1),
            )
        def forward(self, x):
            out, (hn, cn) = self.lstm(x)
            return self.head(hn[-1]).squeeze(-1)
    def explain_prediction(model, sequence, top_k=3):
        return {"reasons": ["Fallback explanation"]}

DEFAULT_OUT_DIR = "/home/claude/return_fraud/outputs"
OUT_DIR = os.getenv("OUT_DIR", DEFAULT_OUT_DIR if os.path.exists(DEFAULT_OUT_DIR) else "./outputs")

app = FastAPI(title="Return Fraud - Behavior Risk API")

CONFIG = {}
_model = None
_calibrator = None

if os.path.exists(f"{OUT_DIR}/decision_config.json"):
    with open(f"{OUT_DIR}/decision_config.json") as f:
        CONFIG = json.load(f)

    _model = ReturnFraudLSTM(n_features=CONFIG.get("n_features", N_FEATURES))
    if os.path.exists(f"{OUT_DIR}/cost_aware_lstm.pt"):
        _model.load_state_dict(torch.load(f"{OUT_DIR}/cost_aware_lstm.pt", map_location="cpu"))
    _model.eval()

    if os.path.exists(f"{OUT_DIR}/calibrator.joblib"):
        _calibrator = joblib.load(f"{OUT_DIR}/calibrator.joblib")
else:
    CONFIG = {"seq_len": SEQ_LEN, "n_features": N_FEATURES, "low_thresh": 0.30, "high_thresh": 0.70}


class Order(BaseModel):
    order_date: str
    category: str
    order_value: float
    returned: int
    return_date: str | None = None
    return_reason: str | None = None
    delivery_confirmed: int = 1


class ScoreRequest(BaseModel):
    customer_id: str
    orders: list[Order]  # chronological order, last item = the return being scored


def build_live_sequence(orders: list[dict]) -> np.ndarray:
    """
    Same feature logic as features.py's build_sequences() (including the
    running/aggregate features), but for a single customer's live history
    instead of a labeled training dataframe -- no is_fraud label needed,
    that's what we're about to predict.
    """
    df = pd.DataFrame(orders)
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
    df["return_date"] = pd.to_datetime(df["return_date"], format="mixed")
    df["high_value_cat"] = df["category"].isin(["apparel_event", "electronics"]).astype(int)
    df["log_order_value"] = np.log1p(df["order_value"])

    step_feats = []
    prev_date = None
    total_orders = 0
    total_returns = 0
    total_spend = 0.0
    total_refunded = 0.0
    category_return_counts = {}
    last_return_date = None

    for _, row in df.iterrows():
        days_since_prev = (row["order_date"] - prev_date).days if prev_date is not None else 0
        prev_date = row["order_date"]

        returned = int(row["returned"])
        days_to_return = 0
        if returned and pd.notna(row["return_date"]):
            days_to_return = (row["return_date"] - row["order_date"]).days

        base_vec = [
            min(days_since_prev, 90) / 90.0,
            row["log_order_value"] / 8.0,
            row["high_value_cat"],
            returned,
            min(days_to_return, 30) / 30.0,
            row["delivery_confirmed"],
        ]
        reason = row["return_reason"] if returned else "none"
        reason_onehot = encode_reason(reason)
        reason_encoded = reason_onehot.index(1) / float(len(REASON_VOCAB))

        rolling_return_rate = (total_returns / total_orders) if total_orders > 0 else 0.0
        rolling_refund_to_spend = (total_refunded / total_spend) if total_spend > 0 else 0.0
        if total_returns > 0:
            category_concentration = max(category_return_counts.values()) / total_returns
        else:
            category_concentration = 0.0
        if last_return_date is not None:
            days_since_last_return = (row["order_date"] - last_return_date).days
            days_since_last_return_norm = min(days_since_last_return, 180) / 180.0
        else:
            days_since_last_return_norm = 1.0

        full_vec = base_vec + [
            reason_encoded,
            rolling_return_rate,
            rolling_refund_to_spend,
            category_concentration,
            days_since_last_return_norm,
        ]
        step_feats.append(full_vec)

        # update running state with THIS order's outcome, for the next iteration
        total_orders += 1
        total_spend += row["order_value"]
        if returned:
            total_returns += 1
            total_refunded += row["order_value"]
            category_return_counts[row["category"]] = category_return_counts.get(row["category"], 0) + 1
            last_return_date = row["return_date"] if pd.notna(row["return_date"]) else row["order_date"]

    seq_len = CONFIG.get("seq_len", SEQ_LEN)
    n_feats = CONFIG.get("n_features", N_FEATURES)

    window = step_feats[-seq_len:]
    pad_len = seq_len - len(window)
    if pad_len > 0:
        window = [[0.0] * n_feats for _ in range(pad_len)] + window

    return np.array(window, dtype=np.float32)


def three_tier(prob: float) -> str:
    low = CONFIG.get("low_thresh", 0.30)
    high = CONFIG.get("high_thresh", 0.70)
    if prob < low:
        return "auto_approve"
    if prob < high:
        return "soft_friction"
    return "deny"


@app.post("/score")
def score(req: ScoreRequest):
    orders = [o.model_dump() for o in req.orders]
    if not orders or orders[-1]["returned"] != 1:
        return {"error": "the last order in 'orders' must be the return event being evaluated (returned=1)"}

    x = build_live_sequence(orders)

    if _model is not None:
        with torch.no_grad():
            raw_logit = _model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            raw_prob = torch.sigmoid(raw_logit).item()
    else:
        raw_prob = 0.15

    if _calibrator is not None:
        calibrated_prob = float(_calibrator.predict_proba(np.array([[raw_prob]]))[0, 1])
    else:
        calibrated_prob = raw_prob

    tier = three_tier(calibrated_prob)
    
    if _model is not None:
        explanation = explain_prediction(_model, x)
    else:
        explanation = {"summary": "Model uninitialized; returning placeholder risk score.", "reasons": ["Model uninitialized."]}

    return {
        "customer_id": req.customer_id,
        "raw_score": round(raw_prob, 4),
        "calibrated_risk_score": round(float(calibrated_prob), 4),
        "decision_tier": tier,
        "explanation": explanation,
    }


@app.get("/health")
def health():
    return {
        "status": "ok", 
        "seq_len": CONFIG.get("seq_len", SEQ_LEN), 
        "n_features": CONFIG.get("n_features", N_FEATURES),
        "artifacts_loaded": _model is not None and _calibrator is not None
    }
