import glob
import json
import os

import numpy as np
import torch
from PIL import Image

import open_clip

STOCK_REFERENCE_DIR = "/home/claude/return_fraud/image_store/stock_reference"
FRAUD_EMBEDDING_STORE = "/home/claude/return_fraud/image_store/fraud_embeddings.json"

_model = None
_preprocess = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    \"\"\"Lazy-loads CLIP once per process -- this is the expensive step (model
    download + weight load), so every function below reuses this singleton
    rather than reloading per request.\"\"\"
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _model.to(_device).eval()
    return _model, _preprocess


def embed_image(image_path: str) -> np.ndarray:
    model, preprocess = _load_model()
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = model.encode_image(img)
        features = features / features.norm(dim=-1, keepdim=True)  # L2-normalize for cosine similarity
    return features.cpu().numpy()[0]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # already L2-normalized, so dot product = cosine similarity


def check_stock_similarity(embedding: np.ndarray, threshold: float = 0.90) -> dict:
    \"\"\"
    Compares against a small local folder of common stock/product photos.
    In production this would be a proper vector index (FAISS) over a much
    larger reference set, or an actual reverse-image-search API -- this is
    the from-scratch version for a hackathon-scale reference set.
    \"\"\"
    if not os.path.isdir(STOCK_REFERENCE_DIR):
        return {"stock_risk_score": 0.0, "best_match": None, "reasons": []}

    best_sim = 0.0
    best_match = None
    for path in glob.glob(f"{STOCK_REFERENCE_DIR}/*"):
        ref_embedding = embed_image(path)
        sim = _cosine_sim(embedding, ref_embedding)
        if sim > best_sim:
            best_sim = sim
            best_match = os.path.basename(path)

    is_match = best_sim >= threshold
    stock_risk_score = max(0.0, (best_sim - threshold) / (1 - threshold)) if is_match else 0.0

    reasons = []
    if is_match:
        reasons.append(f"image closely matches a known stock/reference photo ({best_match}), not an original photo of the item")

    return {"stock_risk_score": stock_risk_score, "best_match": best_match if is_match else None, "reasons": reasons}


def check_fraud_ring_similarity(embedding: np.ndarray, threshold: float = 0.92) -> dict:
    \"\"\"
    Compares against embeddings of images from PAST claims that were
    confirmed fraudulent. A close match across different claim IDs is a
    strong signal -- either the same fraudster reusing an image, or a
    coordinated ring sharing assets.
    \"\"\"
    if not os.path.exists(FRAUD_EMBEDDING_STORE):
        return {"fraud_reuse_risk_score": 0.0, "matched_claim_id": None, "reasons": []}

    with open(FRAUD_EMBEDDING_STORE) as f:
        store = json.load(f)  # claim_id -> list[float]

    best_sim = 0.0
    best_claim = None
    for claim_id, stored_vec in store.items():
        sim = _cosine_sim(embedding, np.array(stored_vec, dtype=np.float32))
        if sim > best_sim:
            best_sim = sim
            best_claim = claim_id

    is_match = best_sim >= threshold
    fraud_reuse_risk_score = max(0.0, (best_sim - threshold) / (1 - threshold)) if is_match else 0.0

    reasons = []
    if is_match:
        reasons.append(f"image closely matches a photo used in a previously confirmed fraud case ({best_claim})")

    return {"fraud_reuse_risk_score": fraud_reuse_risk_score, "matched_claim_id": best_claim if is_match else None, "reasons": reasons}


def store_fraud_embedding(claim_id: str, embedding: np.ndarray):
    \"\"\"Call this once a claim is CONFIRMED fraudulent (human/analyst-labeled),
    so future claims can be checked against it. Do not call this for
    unconfirmed/pending claims -- that would poison the reference set.\"\"\"
    os.makedirs(os.path.dirname(FRAUD_EMBEDDING_STORE), exist_ok=True)
    store = {}
    if os.path.exists(FRAUD_EMBEDDING_STORE):
        with open(FRAUD_EMBEDDING_STORE) as f:
            store = json.load(f)
    store[claim_id] = embedding.tolist()
    with open(FRAUD_EMBEDDING_STORE, "w") as f:
        json.dump(store, f)
