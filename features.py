"""
features.py
------------
Turns raw orders.csv into fixed-length sequences suitable for an LSTM.

For every RETURN event, we build a sequence of that customer's order history
UP TO AND INCLUDING that return (max length SEQ_LEN, left-padded if shorter).
The label is whether that specific return was fraudulent.

This mirrors the real system design: "given everything we know about this
customer's history so far, how risky is THIS return?"
"""

import numpy as np
import pandas as pd

SEQ_LEN = 8  # how many past orders the LSTM looks at
REASON_VOCAB = ["none", "wrong_size", "changed_mind", "wrong_item", "damaged", "item_not_received"]

# Feature 0-6: per-order features (unchanged).
# Feature 7-10: running/aggregate features computed from the customer's history
# STRICTLY BEFORE the current order -- this is what lets the model see "this
# customer already returns a lot" / "concentrated in one category" directly,
# instead of having to infer it from 8 raw rows. Computed before-the-fact (not
# including the current row's own outcome) so a return event's rolling_return_rate
# isn't trivially a function of its own returned_flag.
N_FEATURES = 11


def encode_reason(reason):
    vec = [0] * len(REASON_VOCAB)
    idx = REASON_VOCAB.index(reason) if reason in REASON_VOCAB else 0
    vec[idx] = 1
    return vec


def build_sequences(df):
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
    df["return_date"] = pd.to_datetime(df["return_date"], format="mixed")
    df["high_value_cat"] = df["category"].isin(["apparel_event", "electronics"]).astype(int)
    df["log_order_value"] = np.log1p(df["order_value"])

    sequences = []
    labels = []
    meta = []  # keep customer_id, order_id, customer_type for later analysis (NOT fed to model)

    for cid, group in df.groupby("customer_id"):
        group = group.sort_values("order_date").reset_index(drop=True)
        prev_date = None
        # per-order feature vector (base 3 numeric + returned flag + days_to_return)
        step_feats = []

        # --- running state, updated AFTER each order (so a row's features
        # reflect history strictly BEFORE that order, never its own outcome) ---
        total_orders = 0
        total_returns = 0
        total_spend = 0.0
        total_refunded = 0.0
        category_return_counts = {}
        last_return_date = None

        for i, row in group.iterrows():
            days_since_prev = (row["order_date"] - prev_date).days if prev_date is not None else 0
            prev_date = row["order_date"]

            returned = int(row.get("returned", 0))
            days_to_return = 0
            if returned and pd.notna(row.get("return_date")):
                days_to_return = (row["return_date"] - row["order_date"]).days

            base_vec = [
                min(days_since_prev, 90) / 90.0,        # recency, capped+normalized
                row["log_order_value"] / 8.0,             # normalized order value
                row["high_value_cat"],                     # is high-value category
                returned,                                   # was this order returned
                min(days_to_return, 30) / 30.0,            # how fast it was returned
                row.get("delivery_confirmed", 1),           # delivery status
            ]
            reason = row.get("return_reason") if returned else "none"
            reason_onehot = encode_reason(reason)
            reason_encoded = reason_onehot.index(1) / len(REASON_VOCAB)  # compact reason encoding

            # --- running/aggregate features (state BEFORE this order) ---
            rolling_return_rate = (total_returns / total_orders) if total_orders > 0 else 0.0
            rolling_refund_to_spend = (total_refunded / total_spend) if total_spend > 0 else 0.0
            if total_returns > 0:
                max_cat_returns = max(category_return_counts.values())
                category_concentration = max_cat_returns / total_returns
            else:
                category_concentration = 0.0
            if last_return_date is not None:
                days_since_last_return = (row["order_date"] - last_return_date).days
                days_since_last_return_norm = min(days_since_last_return, 180) / 180.0
            else:
                days_since_last_return_norm = 1.0  # no prior return yet -> treat as "far"

            full_vec = base_vec + [
                reason_encoded,
                rolling_return_rate,
                rolling_refund_to_spend,
                category_concentration,
                days_since_last_return_norm,
            ]

            step_feats.append(full_vec)

            # Only create a training sample when THIS order is a return
            # (that's the moment we'd be scoring "is this return fraudulent")
            if returned:
                window = step_feats[-SEQ_LEN:]
                pad_len = SEQ_LEN - len(window)
                if pad_len > 0:
                    window = [[0.0] * N_FEATURES for _ in range(pad_len)] + window

                sequences.append(window)
                labels.append(int(row.get("is_fraud", 0)))
                meta.append({
                    "customer_id": cid,
                    "order_id": row.get("order_id"),
                    "customer_type": row.get("customer_type", "genuine"),
                    "category": row.get("category"),
                    "order_value": row.get("order_value"),
                    "return_reason": reason,
                    "device_id": row.get("device_id"),
                    "address_id": row.get("address_id"),
                    "is_fraud": int(row.get("is_fraud", 0)),
                })

            # --- NOW update running state with this order's own outcome,
            # so it's available for the NEXT order's features ---
            total_orders += 1
            total_spend += row["order_value"]
            if returned:
                total_returns += 1
                total_refunded += row["order_value"]
                category_return_counts[row["category"]] = category_return_counts.get(row["category"], 0) + 1
                last_return_date = row["return_date"] if pd.notna(row.get("return_date")) else row["order_date"]

    X = np.array(sequences, dtype=np.float32) if sequences else np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32)
    y = np.array(labels, dtype=np.float32) if labels else np.zeros((0,), dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df


if __name__ == "__main__":
    pass
