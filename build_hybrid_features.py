"""
build_hybrid_features.py
-------------------------
Bridges generate_data_real.py -> features.py.

generate_data_real.py writes orders_hybrid.csv (real UCI genuine customers +
synthetic fraud archetypes). features.py's build_sequences() turns any
orders.csv-shaped dataframe into (X, y, meta) sequence tensors -- it doesn't
care whether the rows came from real or synthetic customers, so we reuse it
as-is and just point it at the hybrid file and save with the _hybrid suffix
that train_hybrid.py expects.

Run this after generate_data_real.py and before train_hybrid.py:
    python generate_data_real.py
    python build_hybrid_features.py
    python train_hybrid.py
"""

import os
import numpy as np
import pandas as pd

try:
    from features import build_sequences
except ImportError:
    def build_sequences(df):
        raise NotImplementedError("features.py module build_sequences function is required.")

DEFAULT_DATA_DIR = "/home/claude/return_fraud/data"
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR if os.path.exists(DEFAULT_DATA_DIR) else "./data")


def main():
    orders_file = f"{DATA_DIR}/orders_hybrid.csv"
    if not os.path.exists(orders_file):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Warning: {orders_file} not found. Please run generate_data_real.py first.")
        return

    df = pd.read_csv(orders_file)
    X, y, meta_df = build_sequences(df)

    print(f"Built {X.shape[0]} return-event samples, sequence shape per sample: {X.shape[1:]}")
    print(f"Fraud rate in samples: {y.mean():.2%}")
    if "customer_type" in meta_df.columns:
        print(f"Genuine (real) samples: {(meta_df['customer_type'] == 'genuine').sum()}")
        print(f"Synthetic fraud-archetype samples: {(meta_df['customer_type'] != 'genuine').sum()}")

    np.save(f"{DATA_DIR}/X_hybrid.npy", X)
    np.save(f"{DATA_DIR}/y_hybrid.npy", y)
    meta_df.to_csv(f"{DATA_DIR}/meta_hybrid.csv", index=False)
    print(f"Saved X_hybrid.npy, y_hybrid.npy, meta_hybrid.csv to {DATA_DIR}")


if __name__ == "__main__":
    main()
