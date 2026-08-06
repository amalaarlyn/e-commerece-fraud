"""
generate_data_real.py
----------------------
Builds the training dataset using REAL transactional data (UCI Online Retail,
541,909 line items / 4,372 real UK customers, Dec 2010-Dec 2011, CC BY 4.0)
as the source of GENUINE customer behavior.

WHY HYBRID (real + synthetic), NOT pure real data:
No public dataset contains verified return-fraud ground truth labels (this is
a well-documented limitation in the literature itself, see PRISMA review
findings on data scarcity). So:
  - Genuine customer behavior  -> real UCI transactions (real humans, real noise)
  - Fraud customer behavior    -> synthetically injected, using the same
                                   literature-grounded archetypes as before
                                   (wardrobing, INR abuse, damage fraud, serial rings)

This hybrid approach is standard practice in fraud-detection research given
the acknowledged absence of labeled fraud datasets, and should be stated
explicitly as a methodology decision (not hidden) in the paper.

Real "returns" are identified via UCI's own convention: InvoiceNo starting
with 'C' = cancellation, paired with negative Quantity.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
np.random.seed(RNG_SEED)

DEFAULT_DATA_DIR = "/home/claude/return_fraud/data"
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR if os.path.exists(DEFAULT_DATA_DIR) else "./data")

RAW_PATH = os.path.join(DATA_DIR, "online_retail_test.csv")
OUT_PATH = os.path.join(DATA_DIR, "orders_hybrid.csv")

CATEGORIES = ["apparel_event", "apparel_casual", "electronics", "home", "beauty", "footwear", "accessories"]
HIGH_VALUE_CATS = {"apparel_event", "electronics"}

FRAUD_INJECTION_RATE = 0.15  # 15% fraud archetypes injected alongside real genuine customers
ARCHETYPE_WEIGHTS = {
    "wardrobing": 0.35,
    "inr_abuser": 0.20,
    "damage_fraud": 0.30,
    "serial_fraud": 0.15,
}


def create_synthetic_retail_fallback(n_customers=50):
    """Generates synthetic genuine transactions if UCI raw CSV is missing."""
    rows = []
    for cid in range(1000, 1000 + n_customers):
        n_orders = np.random.randint(3, 8)
        start_date = datetime(2025, 1, 1)
        for i in range(n_orders):
            inv_date = start_date + timedelta(days=i * 15 + np.random.randint(0, 5))
            is_cancel = "C" if (i > 0 and np.random.rand() < 0.15) else ""
            inv_no = f"{is_cancel}{500000 + cid*10 + i}"
            quantity = -np.random.randint(1, 3) if is_cancel else np.random.randint(1, 5)
            unit_price = round(float(np.random.uniform(5.0, 150.0)), 2)
            rows.append({
                "InvoiceNo": inv_no,
                "StockCode": "85123A",
                "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "Quantity": quantity,
                "InvoiceDate": inv_date.strftime("%m/%d/%Y %H:%M"),
                "UnitPrice": unit_price,
                "CustomerID": float(cid),
                "Country": "United Kingdom"
            })
    return pd.DataFrame(rows)


def load_real_genuine_orders():
    if not os.path.exists(RAW_PATH):
        print(f"Notice: {RAW_PATH} not found. Generating synthetic genuine transactions for testing...")
        df = create_synthetic_retail_fallback()
    else:
        df = pd.read_csv(RAW_PATH, encoding="latin1")

    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    df["is_cancel"] = df["InvoiceNo"].astype(str).str.startswith("C")

    # Aggregate line items -> one row per ORDER (InvoiceNo)
    df["line_total"] = (df["Quantity"] * df["UnitPrice"]).abs()
    
    agg = df.groupby(["CustomerID", "InvoiceNo"]).agg(
        order_date=("InvoiceDate", "min"),
        order_value=("line_total", "sum"),
        is_cancel=("is_cancel", "max"),
    ).reset_index()

    # Keep customers with a reasonable amount of history for sequence modeling
    order_counts = agg.groupby("CustomerID").size()
    keep_customers = order_counts[order_counts >= 3].index
    agg = agg[agg["CustomerID"].isin(keep_customers)].copy()

    agg = agg.sort_values(["CustomerID", "order_date"]).reset_index(drop=True)

    rows = []
    for cid, group in agg.groupby("CustomerID"):
        group = group.sort_values("order_date").reset_index(drop=True)
        for i, r in group.iterrows():
            category = np.random.choice(CATEGORIES)
            returned = int(r["is_cancel"])
            return_date = r["order_date"] + timedelta(days=int(np.random.randint(1, 20))) if returned else pd.NaT
            rows.append({
                "customer_id": f"real_{cid}",
                "customer_type": "genuine",
                "order_id": f"real_{cid}_{i}",
                "order_date": r["order_date"],
                "category": category,
                "order_value": round(float(r["order_value"]), 2),
                "device_id": f"DEV_real_{cid}",
                "address_id": f"ADDR_real_{cid}",
                "returned": returned,
                "return_date": return_date,
                "return_reason": np.random.choice(["wrong_size", "changed_mind", "wrong_item", "damaged"],
                                                    p=[0.4, 0.3, 0.2, 0.1]) if returned else None,
                "delivery_confirmed": 1,
                "is_fraud": 0,
            })
    return pd.DataFrame(rows)


def inject_synthetic_fraud(n_customers, start_id=900000):
    rows = []
    types = np.random.choice(list(ARCHETYPE_WEIGHTS.keys()), size=n_customers, p=list(ARCHETYPE_WEIGHTS.values()))

    for j, archetype in enumerate(types):
        cid = f"synth_{start_id + j}"
        if archetype == "wardrobing":
            n_orders = np.random.randint(5, 20); return_rate = np.random.uniform(0.35, 0.55)
        elif archetype == "inr_abuser":
            n_orders = np.random.randint(6, 18); return_rate = np.random.uniform(0.25, 0.45)
        elif archetype == "damage_fraud":
            n_orders = np.random.randint(6, 18); return_rate = np.random.uniform(0.30, 0.50)
        elif archetype == "serial_fraud":
            n_orders = np.random.randint(10, 30); return_rate = np.random.uniform(0.45, 0.70)

        order_dates = sorted(pd.Timestamp("2011-01-01") + pd.to_timedelta(np.random.uniform(0, 300, n_orders), unit="D"))
        shared_device = f"DEV-{np.random.randint(1, 40)}" if archetype == "serial_fraud" else f"DEV-{cid}"
        shared_address = f"ADDR-{np.random.randint(1, 60)}" if archetype == "serial_fraud" else f"ADDR-{cid}"

        for i, order_date in enumerate(order_dates):
            category = np.random.choice(list(HIGH_VALUE_CATS)) if (archetype == "wardrobing" and np.random.rand() < 0.7) else np.random.choice(CATEGORIES)
            order_value = round(np.random.lognormal(mean=4.0, sigma=0.6), 2)
            if category in HIGH_VALUE_CATS:
                order_value *= 1.8
            is_returned = np.random.rand() < return_rate

            row = {
                "customer_id": cid, "customer_type": archetype, "order_id": f"{cid}-{i}",
                "order_date": order_date, "category": category, "order_value": round(order_value, 2),
                "device_id": shared_device, "address_id": shared_address, "returned": int(is_returned),
                "return_date": pd.NaT, "return_reason": None, "delivery_confirmed": 1, "is_fraud": 0,
            }
            if is_returned:
                if archetype == "wardrobing":
                    d = np.random.randint(1, 6); reason = "changed_mind"; fraud = 1 if category in HIGH_VALUE_CATS else 0
                elif archetype == "inr_abuser":
                    d = np.random.randint(0, 3); reason = "item_not_received"; fraud = 1
                elif archetype == "damage_fraud":
                    d = np.random.randint(2, 10); reason = "damaged"; fraud = 1
                elif archetype == "serial_fraud":
                    d = np.random.randint(1, 12); reason = np.random.choice(["changed_mind", "damaged", "item_not_received", "wrong_item"]); fraud = 1
                row["return_date"] = order_date + timedelta(days=int(d))
                row["return_reason"] = reason
                row["is_fraud"] = fraud
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Loading and aggregating REAL UCI Online Retail data...")
    real_df = load_real_genuine_orders()
    n_real_customers = real_df["customer_id"].nunique()
    print(f"Real genuine customers retained (>=3 orders): {n_real_customers}")
    print(f"Real orders: {len(real_df)}, real returns (cancellations): {real_df['returned'].sum()}")

    n_fraud_customers = int(n_real_customers * FRAUD_INJECTION_RATE / (1 - FRAUD_INJECTION_RATE))
    print(f"\nInjecting {n_fraud_customers} synthetic fraud customers ({FRAUD_INJECTION_RATE:.0%} of final mix)...")
    fraud_df = inject_synthetic_fraud(n_fraud_customers)
    print(f"Synthetic orders: {len(fraud_df)}, synthetic fraudulent returns: {fraud_df['is_fraud'].sum()}")

    combined = pd.concat([real_df, fraud_df], ignore_index=True)
    combined = combined.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\n=== Final hybrid dataset ===")
    print(f"Total customers: {combined['customer_id'].nunique()}")
    print(f"Total orders: {len(combined)}")
    print(f"Total returns: {combined['returned'].sum()}")
    print(f"Total fraudulent returns: {combined['is_fraud'].sum()}")
    print(f"Fraud rate among returns: {combined[combined.returned==1]['is_fraud'].mean():.2%}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
