"""
dual_image_api.py
-----------------
FastAPI endpoint that accepts BOTH a product image and a receipt image,
runs the full fraud-detection pipeline on each, and returns a fused risk score.

Pipeline:
  Product Image → Image forensics (EXIF, ELA, duplicate, manipulation)
  Receipt Image → OCR + field parsing + tampering + cross-validation
  Both         → Fused risk score + combined reasons + decision tier

Run:
    uvicorn dual_image_api:app --reload --port 8001

Example (curl):
    curl -X POST http://localhost:8001/score_dual \
      -F "product_image=@product_photo.jpg" \
      -F "receipt_image=@receipt_photo.jpg" \
      -F "claim_id=CLM_12345" \
      -F "customer_id=CUST_9001" \
      -F "order_date=2026-05-01" \
      -F "return_date=2026-05-15" \
      -F "claim_amount=299.99" \
      -F "claim_product=Wireless Headphones"
"""

import base64
import os
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Receipt processing
from receipt_ocr import analyze_receipt

# Image intelligence (OpenCV-based, no heavy ML deps)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "image_intelligence"))

from image_intelligence.preprocessing import preprocess_image
from image_intelligence.scoring import compute_image_score
from image_intelligence.manipulation_check import analyze_manipulation

app = FastAPI(title="Return Fraud - Dual Image Scoring API")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Weights for fusing product image score + receipt score
# ---------------------------------------------------------------------------
FUSION_WEIGHTS = {
    "product_image": 0.55,  # Product image forensics (manipulation, duplicates)
    "receipt":       0.45,  # Receipt OCR + tampering + cross-validation
}

TIER_LOW = 0.30
TIER_HIGH = 0.65


def three_tier(score: float) -> str:
    if score < TIER_LOW:
        return "auto_approve"
    if score < TIER_HIGH:
        return "soft_friction"
    return "deny"


async def _save_upload(upload: UploadFile, prefix: str) -> str:
    """Saves an uploaded file to disk and returns the path."""
    ext = os.path.splitext(upload.filename or "image.jpg")[1] or ".jpg"
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    content = await upload.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath


# ---------------------------------------------------------------------------
# Main endpoint: score both images
# ---------------------------------------------------------------------------

@app.post("/score_dual")
async def score_dual(
    product_image: UploadFile = File(..., description="Product/damage photo"),
    receipt_image: UploadFile = File(..., description="Purchase receipt photo"),
    claim_id: str = Form(...),
    customer_id: str = Form("CUST_UNKNOWN"),
    order_date: Optional[str] = Form(None),
    return_date: Optional[str] = Form(None),
    claim_amount: Optional[float] = Form(None),
    claim_product: Optional[str] = Form(None),
):
    """
    Accepts a product image and receipt image, runs full analysis, and returns:
      - product_image_analysis: manipulation + similarity scores
      - receipt_analysis: OCR + tampering + cross-validation
      - fused_risk_score: weighted combination
      - decision_tier: auto_approve / soft_friction / deny
      - combined_reasons: all human-readable risk explanations
    """
    # --- Save uploads ---
    product_path = await _save_upload(product_image, "product")
    receipt_path = await _save_upload(receipt_image, "receipt")

    try:
        # ═══════════════════════════════════════════════════
        # 1. PRODUCT IMAGE ANALYSIS
        # ═══════════════════════════════════════════════════
        product_img_data = preprocess_image(product_path)

        # Run manipulation heuristics (edge, noise, ELA)
        manipulation_result = analyze_manipulation(product_img_data)

        # Run full image scoring (includes similarity if gallery exists)
        product_score_result = compute_image_score(
            uploaded_img_data=product_img_data,
            candidate_gallery=None,  # TODO: plug in gallery of past claims
            customer_id=customer_id,
            return_id=claim_id,
            image_id=os.path.basename(product_path),
        )

        product_risk_score = product_score_result.image_score
        product_flags = [
            {"code": f.code, "message": f.message, "severity": f.severity}
            for f in product_score_result.risk_flags
        ]
        product_reasons = [f.message for f in product_score_result.risk_flags]

        # ═══════════════════════════════════════════════════
        # 2. RECEIPT IMAGE ANALYSIS
        # ═══════════════════════════════════════════════════
        receipt_result = analyze_receipt(
            receipt_image_input=receipt_path,
            claim_order_date=order_date,
            claim_amount=claim_amount,
            claim_product=claim_product,
        )

        receipt_risk_score = receipt_result["receipt_risk_score"]
        receipt_reasons = receipt_result["all_reasons"]

        # ═══════════════════════════════════════════════════
        # 3. FUSE SCORES
        # ═══════════════════════════════════════════════════
        fused_score = round(
            FUSION_WEIGHTS["product_image"] * product_risk_score
            + FUSION_WEIGHTS["receipt"] * receipt_risk_score,
            4,
        )
        fused_score = min(fused_score, 1.0)
        tier = three_tier(fused_score)

        combined_reasons = []
        if product_reasons:
            combined_reasons.extend(product_reasons)
        if receipt_reasons and receipt_reasons != ["no receipt anomalies detected"]:
            combined_reasons.extend(receipt_reasons)
        if not combined_reasons:
            combined_reasons = ["no fraud signals detected across product image and receipt"]

        # ═══════════════════════════════════════════════════
        # RESPONSE
        # ═══════════════════════════════════════════════════
        return {
            "claim_id": claim_id,
            "customer_id": customer_id,
            "fused_risk_score": fused_score,
            "decision_tier": tier,
            "combined_reasons": combined_reasons,
            "product_image_analysis": {
                "image_risk_score": product_risk_score,
                "risk_flags": product_flags,
                "manipulation_details": {
                    "edge_analysis": manipulation_result["edge_analysis"],
                    "noise_analysis": manipulation_result["noise_analysis"],
                    "ela_analysis": manipulation_result["ela_analysis"],
                    "overall_manipulation_score": manipulation_result["overall_manipulation_score"],
                },
            },
            "receipt_analysis": {
                "receipt_risk_score": receipt_risk_score,
                "parsed_fields": receipt_result["parsed_fields"],
                "tampering_analysis": receipt_result["tampering_analysis"],
                "cross_validation": receipt_result["cross_validation"],
                "ocr_summary": receipt_result["ocr_result"],
            },
            "fusion_weights": FUSION_WEIGHTS,
        }

    finally:
        # Cleanup temp uploads (optional — keep for audit trail in production)
        for path in [product_path, receipt_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Standalone endpoints for individual image types
# ---------------------------------------------------------------------------

@app.post("/score_product_image")
async def score_product_image(
    product_image: UploadFile = File(...),
    claim_id: str = Form(...),
    customer_id: str = Form("CUST_UNKNOWN"),
):
    """Scores only a product/damage image."""
    product_path = await _save_upload(product_image, "product")
    try:
        img_data = preprocess_image(product_path)
        result = compute_image_score(
            uploaded_img_data=img_data,
            customer_id=customer_id,
            return_id=claim_id,
            image_id=os.path.basename(product_path),
        )
        return result.to_dict()
    finally:
        if os.path.exists(product_path):
            os.remove(product_path)


@app.post("/score_receipt")
async def score_receipt(
    receipt_image: UploadFile = File(...),
    claim_id: str = Form(...),
    order_date: Optional[str] = Form(None),
    claim_amount: Optional[float] = Form(None),
    claim_product: Optional[str] = Form(None),
):
    """Scores only a receipt image (OCR + tampering + cross-validation)."""
    receipt_path = await _save_upload(receipt_image, "receipt")
    try:
        return analyze_receipt(
            receipt_image_input=receipt_path,
            claim_order_date=order_date,
            claim_amount=claim_amount,
            claim_product=claim_product,
        )
    finally:
        if os.path.exists(receipt_path):
            os.remove(receipt_path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "fusion_weights": FUSION_WEIGHTS,
        "tiers": {"low": TIER_LOW, "high": TIER_HIGH},
        "endpoints": ["/score_dual", "/score_product_image", "/score_receipt"],
    }
