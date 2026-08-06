import base64

from fastapi import FastAPI
from pydantic import BaseModel

from image_features import extract_exif, check_exif_consistency, compute_ela, check_duplicate, store_claim_hash
from clip_similarity import embed_image, check_stock_similarity, check_fraud_ring_similarity, store_fraud_embedding

app = FastAPI(title="Return Fraud - Image Forensics Risk API")

WEIGHTS = {
    "exif_risk_score": 0.20,
    "ela_risk_score": 0.30,
    "duplicate_risk_score": 0.20,
    "stock_risk_score": 0.15,
    "fraud_reuse_risk_score": 0.15,
}

TIER_LOW = 0.30
TIER_HIGH = 0.65


class ImageScoreRequest(BaseModel):
    claim_id: str
    image_path: str  # server-side path to the uploaded image (already saved by the upload handler)
    order_date: str
    return_date: str


def fuse_scores(signals: dict) -> float:
    total = sum(signals[key] * weight for key, weight in WEIGHTS.items())
    return round(min(total, 1.0), 4)


def three_tier(score: float) -> str:
    if score < TIER_LOW:
        return "auto_approve"
    if score < TIER_HIGH:
        return "soft_friction"
    return "deny"


@app.post("/score_image")
def score_image(req: ImageScoreRequest):
    all_reasons = []

    # --- Step 2: EXIF ---
    exif = extract_exif(req.image_path)
    exif_result = check_exif_consistency(exif, req.order_date, req.return_date)
    all_reasons += exif_result["reasons"]

    # --- Step 3: ELA ---
    ela_result = compute_ela(req.image_path)
    all_reasons += ela_result["reasons"]

    # --- Step 4: Duplicate check ---
    dup_result = check_duplicate(req.image_path)
    all_reasons += dup_result["reasons"]

    # --- Step 5: CLIP similarity ---
    embedding = embed_image(req.image_path)
    stock_result = check_stock_similarity(embedding)
    fraud_result = check_fraud_ring_similarity(embedding)
    all_reasons += stock_result["reasons"]
    all_reasons += fraud_result["reasons"]

    signals = {
        "exif_risk_score": exif_result["exif_risk_score"],
        "ela_risk_score": ela_result["ela_risk_score"],
        "duplicate_risk_score": dup_result["duplicate_risk_score"],
        "stock_risk_score": stock_result["stock_risk_score"],
        "fraud_reuse_risk_score": fraud_result["fraud_reuse_risk_score"],
    }

    # --- Step 6: Fusion ---
    image_risk_score = fuse_scores(signals)
    tier = three_tier(image_risk_score)

    # Persist this claim's fingerprints so FUTURE claims can be checked
    # against it -- store the pHash always (duplicate detection needs every
    # claim, not just fraud), but only store the CLIP embedding as a fraud
    # reference once an analyst confirms fraud (see store_fraud_embedding's
    # docstring) -- NOT here, automatically, on every submission.
    store_claim_hash(req.claim_id, dup_result["_current_hash"])

    return {
        "claim_id": req.claim_id,
        "image_risk_score": image_risk_score,
        "decision_tier": tier,
        "signal_breakdown": signals,
        "reasons": all_reasons if all_reasons else ["no tamper, duplication, or reuse signals detected"],
        "ela_heatmap_png_base64": base64.b64encode(ela_result["diff_image_png"]).decode("utf-8"),
    }


@app.post("/confirm_fraud/{claim_id}")
def confirm_fraud(claim_id: str, image_path: str):
    """
    Call this from the analyst dashboard when a human confirms a claim was
    fraudulent -- adds its CLIP embedding to the fraud-reuse reference set
    used by check_fraud_ring_similarity() for all FUTURE claims.
    """
    embedding = embed_image(image_path)
    store_fraud_embedding(claim_id, embedding)
    return {"status": "stored", "claim_id": claim_id}


@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS, "tiers": {"low": TIER_LOW, "high": TIER_HIGH}}
