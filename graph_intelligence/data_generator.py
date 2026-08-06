"""
data_generator.py

Generates synthetic customer + attribute data since no public real-world
return-fraud-ring dataset exists (this data is proprietary for obvious
reasons). We generate:

  - N legitimate customers, each with mostly independent random attributes
    (occasional harmless overlap, e.g. two strangers on the same ISP IP)
  - M injected "fraud rings": small clusters of customers that deliberately
    reuse 2-3 attributes among themselves

Ground truth (which customers belong to an injected ring) is kept alongside
the data, so we can later compute precision/recall for the evaluation
section of the project.
"""

import random
import uuid
from dataclasses import dataclass, field


def _short_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class CustomerRecord:
    customer_id: str
    device: str
    address: str
    phone: str
    payment: str
    ip: str
    is_fraud_ring_member: bool = False
    ring_id: str = None


def generate_dataset(
    n_legit_customers: int = 200,
    n_rings: int = 5,
    ring_size_range: tuple = (3, 6),
    seed: int = 42,
):
    """
    Returns:
        customers: list[CustomerRecord]
        ground_truth: dict[customer_id -> ring_id or None]
    """
    rng = random.Random(seed)
    customers = []

    # --- Legitimate customers: independent random attributes ---
    for _ in range(n_legit_customers):
        customers.append(
            CustomerRecord(
                customer_id=_short_id("cust"),
                device=_short_id("device"),
                address=_short_id("addr"),
                phone=_short_id("phone"),
                payment=_short_id("card"),
                ip=_short_id("ip"),
            )
        )

    # Introduce a small amount of harmless IP overlap (e.g. shared office/wifi)
    # to make sure our algorithm doesn't over-flag on IP alone.
    if len(customers) >= 4:
        shared_ip = _short_id("ip")
        for c in rng.sample(customers, k=min(4, len(customers))):
            c.ip = shared_ip

    # --- Fraud rings: deliberately share 2-3 attributes within the ring ---
    for _ in range(n_rings):
        ring_id = _short_id("ring")
        ring_size = rng.randint(*ring_size_range)

        shared_device = _short_id("device")
        shared_address = _short_id("addr")
        shared_payment = _short_id("card")

        # randomly decide which 2-3 attributes this particular ring reuses,
        # so not every ring looks identical
        shared_attrs = rng.sample(
            ["device", "address", "payment"], k=rng.randint(2, 3)
        )

        for _ in range(ring_size):
            customers.append(
                CustomerRecord(
                    customer_id=_short_id("cust"),
                    device=shared_device if "device" in shared_attrs else _short_id("device"),
                    address=shared_address if "address" in shared_attrs else _short_id("addr"),
                    phone=_short_id("phone"),
                    payment=shared_payment if "payment" in shared_attrs else _short_id("card"),
                    ip=_short_id("ip"),
                    is_fraud_ring_member=True,
                    ring_id=ring_id,
                )
            )

    rng.shuffle(customers)

    ground_truth = {
        c.customer_id: c.ring_id for c in customers if c.is_fraud_ring_member
    }

    return customers, ground_truth


if __name__ == "__main__":
    customers, ground_truth = generate_dataset()
    print(f"Generated {len(customers)} customers")
    print(f"Fraud ring members: {len(ground_truth)}")
    print("Sample record:", customers[0])
