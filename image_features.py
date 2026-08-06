import io
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image, ExifTags

# Where perceptual hashes of past claim images are stored, so new
# submissions can be checked for reuse against everything seen before.
HASH_STORE_PATH = "/home/claude/return_fraud/image_store/claim_hashes.json"

EDITING_SOFTWARE_MARKERS = [
    "photoshop", "gimp", "lightroom", "snapseed", "picsart", "facetune",
]


def extract_exif(image_path: str) -> dict:
    \"\"\"
    Pulls capture timestamp, camera make/model, and editing-software tag.
    Returns a dict with None for any field the image doesn't carry
    (screenshots and many messaging-app-compressed photos strip EXIF
    entirely -- that absence is itself a (weak) signal, handled in scoring).
    \"\"\"
    img = Image.open(image_path)
    exif_raw = img.getexif()

    if not exif_raw:
        return {"has_exif": False, "capture_datetime": None, "camera": None, "software": None}

    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}

    capture_dt = None
    for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
        if key in tags:
            try:
                capture_dt = datetime.strptime(tags[key], "%Y:%m:%d %H:%M:%S")
                break
            except (ValueError, TypeError):
                continue

    camera = None
    if "Make" in tags or "Model" in tags:
        camera = f"{tags.get('Make', '')} {tags.get('Model', '')}".strip()

    software = tags.get("Software")

    return {
        "has_exif": True,
        "capture_datetime": capture_dt.isoformat() if capture_dt else None,
        "camera": camera,
        "software": software,
    }


def check_exif_consistency(exif: dict, order_date: str, return_date: str) -> dict:
    \"\"\"
    Compares the EXIF capture date against the claim's actual order/return
    window, and flags known editing-software fingerprints in the tag itself.
    \"\"\"
    reasons = []
    risk_signals = 0

    if exif["software"]:
        sw_lower = exif["software"].lower()
        if any(marker in sw_lower for marker in EDITING_SOFTWARE_MARKERS):
            reasons.append(f"image metadata shows it was processed in {exif['software']}")
            risk_signals += 1

    if exif["capture_datetime"]:
        capture = datetime.fromisoformat(exif["capture_datetime"])
        order = datetime.fromisoformat(order_date)
        ret = datetime.fromisoformat(return_date)
        if capture < order:
            reasons.append("photo was taken before the order was even placed")
            risk_signals += 1
        elif capture > ret:
            reasons.append("photo was taken after the return was already filed")
            risk_signals += 1
    elif not exif["has_exif"]:
        # Weak signal on its own -- screenshots and many chat apps strip EXIF
        # for legitimate reasons -- but worth surfacing for the fusion step.
        reasons.append("image has no capture metadata (common for screenshots, not conclusive alone)")

    exif_risk_score = min(risk_signals / 2.0, 1.0)  # 2 strong signals = max risk from this check
    return {"exif_risk_score": exif_risk_score, "reasons": reasons}


def compute_ela(image_path: str, quality: int = 90) -> dict:
    \"\"\"
    Error Level Analysis: recompress the image at a known JPEG quality and
    diff it against the original. Uniform low error = untouched photo.
    Bright localized patches = likely edited/cloned/pasted region.

    Returns a summary score (0-1) plus the diff image bytes (PNG) so the
    API layer can optionally return a visual heatmap to the analyst UI.
    \"\"\"
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    orig_arr = np.asarray(original, dtype=np.int16)
    recompressed_arr = np.asarray(recompressed, dtype=np.int16)
    diff = np.abs(orig_arr - recompressed_arr)

    diff_gray = diff.mean(axis=2)  # (H, W)

    # A tampered region shows up as a spatial outlier relative to the
    # image's own overall noise floor -- either unusually HIGH local error
    # (a blended/retouched region with mismatched compression history) or
    # unusually LOW local error (a cleanly pasted flat patch that barely
    # changes under recompression while the surrounding photo noise does).
    # We flag whichever direction deviates most, using absolute z-score.
    global_mean = diff_gray.mean()
    # Absolute floor on std, not just a numerical epsilon: a flat/plain
    # background (very common in product return photos) can have near-zero
    # natural compression variance, which would otherwise make the z-score
    # blow up on trivial noise unrelated to tampering. 2.0 is in raw pixel
    # error units (0-255 scale), not normalized -- calibrate this against
    # your own real photo set once you have one.
    global_std = max(diff_gray.std(), 2.0)
    block = 16
    h, w = diff_gray.shape
    max_abs_zscore = 0.0
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            local_mean = diff_gray[y:y + block, x:x + block].mean()
            z = abs(local_mean - global_mean) / global_std
            max_abs_zscore = max(max_abs_zscore, z)

    # Normalize: |z| above ~3 is a strong outlier in a natural photo's own
    # recompression-error distribution
    ela_risk_score = float(min(max_abs_zscore / 3.0, 1.0))

    diff_img = Image.fromarray((diff_gray / (diff_gray.max() + 1e-6) * 255).astype(np.uint8))
    diff_buffer = io.BytesIO()
    diff_img.save(diff_buffer, "PNG")

    reasons = []
    if ela_risk_score > 0.5:
        reasons.append("recompression analysis found a localized region that decays differently from the rest of the image, consistent with editing")

    return {"ela_risk_score": ela_risk_score, "diff_image_png": diff_buffer.getvalue(), "reasons": reasons}


def compute_phash(image_path: str, hash_size: int = 8) -> str:
    \"\"\"
    Perceptual hash via DCT, implemented from scratch (no imagehash
    dependency) -- resize to (hash_size*4)^2 grayscale, take the low-frequency
    DCT coefficients, and threshold against the median to get a binary
    fingerprint that's stable under recompression/resizing but sensitive to
    genuine content changes.
    \"\"\"
    from scipy.fftpack import dct

    img = Image.open(image_path).convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)

    dct_full = dct(dct(pixels, axis=0), axis=1)
    dct_low = dct_full[:hash_size, :hash_size]

    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return sum(a != b for a, b in zip(hash_a, hash_b))


def check_duplicate(image_path: str, similarity_threshold: int = 6) -> dict:
    \"\"\"
    Compares this image's pHash against every previously stored claim hash.
    A small Hamming distance (few bits differ) means the same photo -- or a
    lightly re-edited version of it -- was submitted before, possibly on a
    DIFFERENT claim/customer (a fraud-ring or repeat-fraud signal).
    \"\"\"
    current_hash = compute_phash(image_path)

    os.makedirs(os.path.dirname(HASH_STORE_PATH), exist_ok=True)
    if os.path.exists(HASH_STORE_PATH):
        with open(HASH_STORE_PATH) as f:
            store = json.load(f)
    else:
        store = {}  # claim_id -> phash

    best_match_claim = None
    best_distance = len(current_hash) + 1
    for claim_id, stored_hash in store.items():
        d = hamming_distance(current_hash, stored_hash)
        if d < best_distance:
            best_distance = d
            best_match_claim = claim_id

    is_duplicate = best_match_claim is not None and best_distance <= similarity_threshold
    duplicate_risk_score = max(0.0, 1.0 - (best_distance / (similarity_threshold * 2))) if is_duplicate else 0.0

    reasons = []
    if is_duplicate:
        reasons.append(f"this image (or a near copy of it) matches a previous claim ({best_match_claim})")

    return {
        "duplicate_risk_score": duplicate_risk_score,
        "matched_claim_id": best_match_claim if is_duplicate else None,
        "hamming_distance": best_distance if is_duplicate else None,
        "reasons": reasons,
        "_current_hash": current_hash,  # returned so the caller can store it after scoring
    }


def store_claim_hash(claim_id: str, phash: str):
    \"\"\"Call this AFTER scoring a claim, so future claims can be checked against it.\"\"\"
    os.makedirs(os.path.dirname(HASH_STORE_PATH), exist_ok=True)
    store = {}
    if os.path.exists(HASH_STORE_PATH):
        with open(HASH_STORE_PATH) as f:
            store = json.load(f)
    store[claim_id] = phash
    with open(HASH_STORE_PATH, "w") as f:
        json.dump(store, f)
