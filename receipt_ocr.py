"""
receipt_ocr.py
--------------
Receipt image processing module for the Return Fraud Detection Platform.

Capabilities:
  1. OCR extraction via Tesseract — pulls raw text from receipt images.
  2. Structured field parsing — merchant name, date, total amount, line items,
     payment method, transaction/receipt number.
  3. Receipt tampering detection — font consistency, alignment regularity,
     compression artifact anomalies specific to receipt images.
  4. Cross-validation — compares extracted receipt data against the claim's
     stated order date, amount, and product category to surface mismatches.

Dependencies:
  - pytesseract (pip install pytesseract)
  - tesseract binary (brew install tesseract on macOS)
  - PIL / Pillow
  - OpenCV (cv2)
  - numpy
"""

import io
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

try:
    import pytesseract
except ImportError:
    pytesseract = None


# ---------------------------------------------------------------------------
# OCR Preprocessing — enhance receipt image for better Tesseract accuracy
# ---------------------------------------------------------------------------

def _preprocess_for_ocr(image_input) -> np.ndarray:
    """
    Converts input to a high-contrast grayscale image optimized for OCR.
    Receipts are typically low-contrast thermal prints, so we apply:
      - Grayscale conversion
      - Adaptive thresholding (handles uneven lighting)
      - Noise removal via morphological operations
      - Deskewing if the receipt is slightly rotated
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Cannot read image at: {image_input}")
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=12)

    # Adaptive threshold — handles thermal receipt's uneven contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # Morphological close to connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned


def _deskew(image: np.ndarray) -> np.ndarray:
    """Corrects slight rotation in scanned/photographed receipts."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 50:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ---------------------------------------------------------------------------
# OCR Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_receipt(image_input, lang: str = "eng") -> Dict[str, Any]:
    """
    Runs Tesseract OCR on a receipt image and returns:
      - raw_text: full extracted text
      - lines: list of non-empty text lines
      - confidence: average word confidence (0-100)
      - word_details: per-word bounding boxes and confidence (for tampering analysis)
    """
    if pytesseract is None:
        return {
            "raw_text": "",
            "lines": [],
            "confidence": 0.0,
            "word_details": [],
            "error": "pytesseract not installed. Run: pip install pytesseract",
        }

    preprocessed = _preprocess_for_ocr(image_input)
    deskewed = _deskew(preprocessed)

    # Get detailed word-level data (bounding boxes + confidence)
    detail_data = pytesseract.image_to_data(
        deskewed, lang=lang, output_type=pytesseract.Output.DICT
    )

    word_details = []
    confidences = []
    for i in range(len(detail_data["text"])):
        word = detail_data["text"][i].strip()
        conf = int(detail_data["conf"][i])
        if word and conf > 0:
            word_details.append({
                "text": word,
                "confidence": conf,
                "x": detail_data["left"][i],
                "y": detail_data["top"][i],
                "w": detail_data["width"][i],
                "h": detail_data["height"][i],
                "block": detail_data["block_num"][i],
                "line": detail_data["line_num"][i],
            })
            confidences.append(conf)

    # Full text extraction
    raw_text = pytesseract.image_to_string(deskewed, lang=lang)
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    return {
        "raw_text": raw_text,
        "lines": lines,
        "confidence": round(avg_confidence, 2),
        "word_details": word_details,
    }


# ---------------------------------------------------------------------------
# Structured Field Parsing
# ---------------------------------------------------------------------------

# Common date formats found on receipts
_DATE_PATTERNS = [
    (r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", "%m/%d/%Y"),  # MM/DD/YYYY
    (r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b", "%Y/%m/%d"),  # YYYY/MM/DD
    (r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b", "%m/%d/%y"),  # MM/DD/YY
    (r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})\b",
     None),  # DD Month YYYY — handled specially
]

# Currency amount patterns
_AMOUNT_PATTERN = re.compile(
    r"(?:(?:TOTAL|AMOUNT|GRAND\s*TOTAL|SUBTOTAL|NET|BALANCE\s*DUE)\s*:?\s*)"
    r"[$€£₹]?\s*(\d{1,6}[.,]\d{2})",
    re.IGNORECASE,
)
_GENERIC_AMOUNT_PATTERN = re.compile(r"[$€£₹]\s*(\d{1,6}[.,]\d{2})")

# Receipt/transaction number
_RECEIPT_NUM_PATTERN = re.compile(
    r"(?:RECEIPT|TRANS(?:ACTION)?|REF|INVOICE|ORDER)\s*(?:#|NO\.?|NUM\.?|:)\s*([A-Z0-9\-]+)",
    re.IGNORECASE,
)

# Payment method
_PAYMENT_PATTERNS = [
    (re.compile(r"\b(VISA|MASTERCARD|AMEX|DISCOVER|DINERS)\b", re.IGNORECASE), "credit_card"),
    (re.compile(r"\bDEBIT\b", re.IGNORECASE), "debit_card"),
    (re.compile(r"\bCASH\b", re.IGNORECASE), "cash"),
    (re.compile(r"\b(APPLE\s*PAY|GOOGLE\s*PAY|SAMSUNG\s*PAY|UPI|PAYPAL)\b", re.IGNORECASE), "digital_wallet"),
]


def _parse_date(text: str) -> Optional[str]:
    """Attempts to extract a date from receipt text."""
    for pattern, fmt in _DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                if fmt:
                    date_str = match.group(0)
                    # Normalize separators
                    date_str = date_str.replace("-", "/").replace(".", "/")
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%Y-%m-%d")
                else:
                    # DD Month YYYY format
                    day, month_str, year = match.groups()
                    dt = datetime.strptime(f"{day} {month_str[:3]} {year}", "%d %b %Y")
                    return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
    return None


def _parse_total_amount(text: str) -> Optional[float]:
    """Extracts the total/grand total amount from receipt text."""
    match = _AMOUNT_PATTERN.search(text)
    if match:
        amount_str = match.group(1).replace(",", ".")
        try:
            return float(amount_str)
        except ValueError:
            pass
    # Fallback: find the largest currency amount on the receipt
    all_amounts = _GENERIC_AMOUNT_PATTERN.findall(text)
    if all_amounts:
        amounts = []
        for a in all_amounts:
            try:
                amounts.append(float(a.replace(",", ".")))
            except ValueError:
                continue
        if amounts:
            return max(amounts)
    return None


def _parse_line_items(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts line items from receipt text.
    Looks for lines matching pattern: item description + price
    """
    items = []
    item_pattern = re.compile(
        r"^(.{3,40}?)\s+[$€£₹]?\s*(\d{1,5}[.,]\d{2})\s*$"
    )
    for line in lines:
        match = item_pattern.match(line.strip())
        if match:
            name = match.group(1).strip()
            # Skip header/footer lines
            skip_keywords = {"total", "subtotal", "tax", "change", "cash", "card",
                             "visa", "receipt", "thank", "date", "time", "store"}
            if any(kw in name.lower() for kw in skip_keywords):
                continue
            try:
                price = float(match.group(2).replace(",", "."))
                items.append({"name": name, "price": price})
            except ValueError:
                continue
    return items


def _parse_merchant(lines: List[str]) -> Optional[str]:
    """
    The merchant name is typically the first 1-2 non-empty lines on a receipt.
    """
    for line in lines[:3]:
        # Skip lines that are just dates, numbers, or very short
        if len(line) < 3:
            continue
        if re.match(r"^[\d\s/\-:.]+$", line):
            continue
        return line.strip()
    return None


def _parse_payment_method(text: str) -> Optional[str]:
    """Detects payment method from receipt text."""
    for pattern, method in _PAYMENT_PATTERNS:
        if pattern.search(text):
            return method
    return None


def parse_receipt_fields(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts structured fields from OCR output.
    Returns a dict with: merchant, date, total, items, receipt_number, payment_method.
    """
    raw_text = ocr_result.get("raw_text", "")
    lines = ocr_result.get("lines", [])

    merchant = _parse_merchant(lines)
    date = _parse_date(raw_text)
    total = _parse_total_amount(raw_text)
    items = _parse_line_items(lines)
    receipt_num_match = _RECEIPT_NUM_PATTERN.search(raw_text)
    receipt_number = receipt_num_match.group(1) if receipt_num_match else None
    payment_method = _parse_payment_method(raw_text)

    return {
        "merchant": merchant,
        "date": date,
        "total_amount": total,
        "line_items": items,
        "receipt_number": receipt_number,
        "payment_method": payment_method,
        "ocr_confidence": ocr_result.get("confidence", 0.0),
        "raw_lines_count": len(lines),
    }


# ---------------------------------------------------------------------------
# Receipt Tampering Detection
# ---------------------------------------------------------------------------

def detect_receipt_tampering(image_input, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes a receipt image for signs of digital tampering:

    1. Font size consistency — legitimate receipts use uniform thermal print;
       edited receipts often have inconsistent character heights.
    2. Alignment regularity — text on real receipts is tightly column-aligned;
       edits break alignment patterns.
    3. ELA (Error Level Analysis) — receipts are typically single-compression
       images; edited regions show compression-level mismatches.
    4. OCR confidence variance — tampered text regions often yield lower
       OCR confidence scores vs. genuine regions.

    Returns a dict with per-check scores and an overall tampering_risk_score.
    """
    reasons = []

    # --- 1. Font/character height consistency ---
    word_details = ocr_result.get("word_details", [])
    font_consistency_score = 0.0
    if len(word_details) >= 10:
        heights = [w["h"] for w in word_details if w["h"] > 5]
        if heights:
            mean_h = np.mean(heights)
            std_h = np.std(heights)
            cv_height = std_h / (mean_h + 1e-5)
            # Thermal receipt printers produce very uniform char heights (CV < 0.25)
            # Edited receipts: different fonts/sizes give CV > 0.35
            if cv_height > 0.35:
                font_consistency_score = min((cv_height - 0.35) / 0.30, 1.0)
                reasons.append(
                    f"character height variation ({cv_height:.2f}) exceeds normal thermal print range, "
                    f"suggesting mixed fonts or digitally edited text"
                )

    # --- 2. Text alignment regularity ---
    alignment_score = 0.0
    if len(word_details) >= 10:
        # Group words by line, check left-margin consistency
        line_groups = {}
        for w in word_details:
            key = (w["block"], w["line"])
            if key not in line_groups:
                line_groups[key] = []
            line_groups[key].append(w)

        left_margins = []
        for words in line_groups.values():
            if words:
                left_margins.append(min(w["x"] for w in words))

        if len(left_margins) >= 5:
            # Real receipts have 1-3 distinct left margins (left-aligned columns)
            unique_margins = len(set(round(m / 10) * 10 for m in left_margins))  # cluster within 10px
            margin_ratio = unique_margins / len(left_margins)
            # High ratio = many different left margins = irregular alignment
            if margin_ratio > 0.5 and unique_margins > 5:
                alignment_score = min((margin_ratio - 0.5) / 0.3, 1.0)
                reasons.append(
                    f"text alignment is irregular ({unique_margins} distinct left margins across "
                    f"{len(left_margins)} lines), unusual for a genuine printed receipt"
                )

    # --- 3. ELA on receipt image ---
    ela_score = 0.0
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        bgr = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        bgr = image_input.copy()
    else:
        bgr = None

    if bgr is not None:
        success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            ela_diff = cv2.absdiff(bgr, recompressed)
            ela_gray = cv2.cvtColor(ela_diff, cv2.COLOR_BGR2GRAY)
            scaled_ela = cv2.convertScaleAbs(ela_gray, alpha=15.0)
            std_delta = float(np.std(scaled_ela))

            if std_delta > 30.0:
                ela_score = min((std_delta - 30.0) / 25.0, 1.0)
                reasons.append(
                    "compression artifact analysis (ELA) detected uneven error levels, "
                    "consistent with post-compression editing of receipt text or amounts"
                )

    # --- 4. OCR confidence variance ---
    confidence_variance_score = 0.0
    if len(word_details) >= 10:
        confs = [w["confidence"] for w in word_details]
        mean_conf = np.mean(confs)
        std_conf = np.std(confs)
        # Legitimate receipts: uniform print quality → low confidence variance
        # Tampered receipts: edited regions have different quality → high variance
        if std_conf > 20.0 and mean_conf < 75.0:
            confidence_variance_score = min((std_conf - 20.0) / 20.0, 1.0)
            reasons.append(
                f"OCR confidence varies significantly across text regions (std={std_conf:.1f}), "
                f"suggesting portions of the receipt were digitally modified"
            )

    # --- Overall tampering score ---
    scores = [font_consistency_score, alignment_score, ela_score, confidence_variance_score]
    # Use weighted max — any single strong signal is sufficient
    overall_score = round(
        0.30 * max(scores) + 0.70 * np.mean(scores), 4
    )
    overall_score = min(overall_score, 1.0)

    return {
        "receipt_tampering_risk_score": overall_score,
        "font_consistency_score": round(font_consistency_score, 4),
        "alignment_regularity_score": round(alignment_score, 4),
        "ela_compression_score": round(ela_score, 4),
        "ocr_confidence_variance_score": round(confidence_variance_score, 4),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Cross-Validation — Receipt Data vs. Claim Metadata
# ---------------------------------------------------------------------------

def cross_validate_receipt(
    receipt_fields: Dict[str, Any],
    claim_order_date: Optional[str] = None,
    claim_amount: Optional[float] = None,
    claim_product: Optional[str] = None,
    date_tolerance_days: int = 3,
    amount_tolerance_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Compares structured receipt data against the claim's stated metadata.
    Flags mismatches between what the receipt says and what the customer claimed.

    Args:
        receipt_fields: Parsed receipt fields from parse_receipt_fields()
        claim_order_date: ISO date string the customer claimed for the order
        claim_amount: Dollar amount the customer claimed
        claim_product: Product name/category the customer claimed
        date_tolerance_days: Acceptable date mismatch window
        amount_tolerance_pct: Acceptable amount deviation (0.10 = 10%)

    Returns:
        Dict with mismatch flags, a cross_validation_risk_score, and reasons.
    """
    mismatches = []
    mismatch_count = 0
    checks_performed = 0

    # --- Date cross-check ---
    if claim_order_date and receipt_fields.get("date"):
        checks_performed += 1
        try:
            claim_dt = datetime.fromisoformat(claim_order_date)
            receipt_dt = datetime.strptime(receipt_fields["date"], "%Y-%m-%d")
            delta_days = abs((claim_dt - receipt_dt).days)
            if delta_days > date_tolerance_days:
                mismatch_count += 1
                mismatches.append(
                    f"receipt date ({receipt_fields['date']}) differs from claimed order date "
                    f"({claim_order_date}) by {delta_days} days (tolerance: {date_tolerance_days})"
                )
        except (ValueError, TypeError):
            pass

    # --- Amount cross-check ---
    if claim_amount is not None and receipt_fields.get("total_amount") is not None:
        checks_performed += 1
        receipt_total = receipt_fields["total_amount"]
        deviation = abs(receipt_total - claim_amount) / max(claim_amount, 0.01)
        if deviation > amount_tolerance_pct:
            mismatch_count += 1
            mismatches.append(
                f"receipt total (${receipt_total:.2f}) differs from claimed amount "
                f"(${claim_amount:.2f}) by {deviation*100:.1f}% (tolerance: {amount_tolerance_pct*100:.0f}%)"
            )

    # --- Product cross-check (fuzzy) ---
    if claim_product and receipt_fields.get("line_items"):
        checks_performed += 1
        item_names = " ".join(item["name"].lower() for item in receipt_fields["line_items"])
        claim_lower = claim_product.lower()
        # Simple keyword overlap check
        claim_words = set(claim_lower.split())
        item_words = set(item_names.split())
        overlap = claim_words & item_words
        if not overlap and len(claim_words) > 0:
            mismatch_count += 1
            mismatches.append(
                f"claimed product '{claim_product}' not found in receipt line items"
            )

    # --- Score ---
    if checks_performed > 0:
        cross_val_score = round(mismatch_count / checks_performed, 4)
    else:
        cross_val_score = 0.0  # Can't validate without data

    return {
        "cross_validation_risk_score": cross_val_score,
        "checks_performed": checks_performed,
        "mismatches_found": mismatch_count,
        "mismatch_details": mismatches,
    }


# ---------------------------------------------------------------------------
# High-Level Pipeline: Full Receipt Analysis
# ---------------------------------------------------------------------------

def analyze_receipt(
    receipt_image_input,
    claim_order_date: Optional[str] = None,
    claim_amount: Optional[float] = None,
    claim_product: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full receipt analysis pipeline:
      1. OCR text extraction
      2. Structured field parsing
      3. Tampering detection
      4. Cross-validation against claim metadata

    Returns a comprehensive result dict with all sub-scores and a fused
    receipt_risk_score.
    """
    # Step 1: OCR
    ocr_result = extract_text_from_receipt(receipt_image_input)

    # Step 2: Parse structured fields
    receipt_fields = parse_receipt_fields(ocr_result)

    # Step 3: Tampering detection
    tampering_result = detect_receipt_tampering(receipt_image_input, ocr_result)

    # Step 4: Cross-validation
    cross_val_result = cross_validate_receipt(
        receipt_fields,
        claim_order_date=claim_order_date,
        claim_amount=claim_amount,
        claim_product=claim_product,
    )

    # --- Fuse into overall receipt risk score ---
    # Weights: tampering is the strongest signal, cross-validation is secondary,
    # low OCR confidence is a weak ambient signal.
    tampering_score = tampering_result["receipt_tampering_risk_score"]
    cross_val_score = cross_val_result["cross_validation_risk_score"]
    confidence_penalty = max(0.0, (50.0 - ocr_result.get("confidence", 50.0)) / 100.0)

    fused_score = round(
        0.50 * tampering_score
        + 0.35 * cross_val_score
        + 0.15 * confidence_penalty,
        4,
    )
    
    # CRITICAL FALLBACK: If no text could be extracted or no structured fields parsed,
    # it implies the image is NOT a receipt (or is completely illegible).
    # This must be heavily penalized to avoid Auto-Approving non-receipt images.
    is_empty_receipt = (
        len(ocr_result.get("raw_text", "").strip()) < 5 
        or cross_val_result["checks_performed"] == 0
    )
    
    if is_empty_receipt:
        fused_score = max(fused_score, 0.85)
        cross_val_result["mismatch_details"].append(
            "CRITICAL: No recognizable receipt data, text, or matching fields could be found in the image. Image may not be a receipt."
        )

    fused_score = min(fused_score, 1.0)

    all_reasons = tampering_result["reasons"] + cross_val_result["mismatch_details"]
    if confidence_penalty > 0.2:
        all_reasons.append(
            f"overall OCR confidence is low ({ocr_result['confidence']:.0f}%), "
            f"receipt may be blurry, damaged, or synthetically generated"
        )

    return {
        "receipt_risk_score": fused_score,
        "ocr_result": {
            "raw_text": ocr_result["raw_text"],
            "lines": ocr_result["lines"],
            "confidence": ocr_result["confidence"],
            "word_count": len(ocr_result.get("word_details", [])),
        },
        "parsed_fields": receipt_fields,
        "tampering_analysis": tampering_result,
        "cross_validation": cross_val_result,
        "all_reasons": all_reasons if all_reasons else ["no receipt anomalies detected"],
    }
