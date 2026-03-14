#!/usr/bin/env python3
# ============================================================
# TWO-LAYER PHISHING DETECTION API
# Layer 1: TLS/URL tabular features  (HTTPS only)
# Layer 2: HTML/DOM + URL heuristics (always runs)
# ============================================================

import os
import re
import pickle
import logging
import numpy as np
import requests
import scipy.sparse as sp

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import HashingVectorizer

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import your existing Layer 1 helpers ─────────────────
from tls_capture import TLSTrafficCapture
from feature_extractor import extract_features_from_url_and_xml

from typing import Optional


# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
LAYER1_MODEL_PATH     = "Model/Layer-1/layer1_model.keras"
LAYER1_SCALER_PATH    = "Model/Layer-1/layer1_scaler.pkl"
LAYER1_FEATURES_PATH  = "Model/Layer-1/layer1_features.pkl"
LAYER1_THRESHOLD_PATH = "Model/Layer-1/layer1_threshold.pkl"

LAYER2_MODEL_PATH     = "Model/Layer-2/layer2_model.keras"
LAYER2_TFIDF_PATH     = "Model/Layer-2/layer2_tfidf.pkl"
LAYER2_SVD_PATH       = "Model/Layer-2/layer2_svd.pkl"
LAYER2_SCALER_PATH    = "Model/Layer-2/layer2_url_scaler.pkl"
LAYER2_THRESHOLD_PATH = "Model/Layer-2/layer2_threshold.pkl"

# HTML feature extraction config — must match training
MAX_CHARS  = 3000
N_FEATURES = 2**15
NGRAMS     = (5, 5)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# LOAD ALL MODELS AT STARTUP
# ─────────────────────────────────────────────────────────
logger.info("Loading Layer 1 artifacts...")
l1_model     = tf.keras.models.load_model(LAYER1_MODEL_PATH, compile=False)
l1_scaler    = pickle.load(open(LAYER1_SCALER_PATH,    "rb"))
l1_features  = pickle.load(open(LAYER1_FEATURES_PATH,  "rb"))
l1_threshold = pickle.load(open(LAYER1_THRESHOLD_PATH, "rb"))
logger.info(f"Layer 1 loaded — threshold={l1_threshold}, features={len(l1_features)}")

logger.info("Loading Layer 2 artifacts...")
l2_model     = tf.keras.models.load_model(LAYER2_MODEL_PATH, compile=False)
l2_tfidf     = pickle.load(open(LAYER2_TFIDF_PATH,    "rb"))
l2_svd       = pickle.load(open(LAYER2_SVD_PATH,      "rb"))
l2_scaler    = pickle.load(open(LAYER2_SCALER_PATH,   "rb"))
l2_threshold = pickle.load(open(LAYER2_THRESHOLD_PATH,"rb"))
logger.info(f"Layer 2 loaded — threshold={l2_threshold}, svd_dim={l2_svd.n_components}")

# Initialize TLS capturer once
capturer = TLSTrafficCapture()

# HashingVectorizer — stateless, recreate with same config as training
hv = HashingVectorizer(
    analyzer="char_wb", ngram_range=NGRAMS,
    n_features=N_FEATURES, alternate_sign=False,
    norm=None, dtype=np.float32
)

# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(title="Two-Layer Phishing Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

# ─────────────────────────────────────────────────────────
# HELPER: HTML CLEANING  (same as training)
# ─────────────────────────────────────────────────────────
def clean_html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    txt = soup.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).lower()
    return txt[:MAX_CHARS]

# ─────────────────────────────────────────────────────────
# HELPER: URL HEURISTIC FEATURES  (same 20 as training)
# ─────────────────────────────────────────────────────────
def extract_url_features(url: str) -> np.ndarray:
    if not isinstance(url, str) or url.strip() == "":
        return np.zeros(20, dtype=np.float32)
    try:
        parsed   = urlparse(url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        query    = parsed.query or ""
        full_url = url.lower()
    except:
        return np.zeros(20, dtype=np.float32)

    ip_pat    = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    sus_words = ["secure","login","update","verify","account","banking",
                 "confirm","paypal","signin","ebay","amazon","free",
                 "lucky","password","credential"]
    sus_tlds  = [".tk",".ml",".ga",".cf",".gq",".xyz",
                 ".top",".club",".online",".site",".info"]

    feats = [
        float(len(url)),
        float(len(hostname)),
        float(len(path)),
        float(len(query)),
        float(url.count(".")),
        float(hostname.count(".")),
        float(max(0, hostname.count(".") - 1)),
        float("@" in url),
        float("//" in path),
        float(url.count("-")),
        float(url.count("_")),
        float(url.count("?")),
        float(url.count("=")),
        float(url.count("&")),
        float(url.count("%")),
        float(parsed.scheme == "https"),
        float(parsed.scheme == "http"),
        float(bool(ip_pat.match(hostname))),
        float(sum(1 for w in sus_words if w in full_url)),
        float(any(hostname.endswith(t) for t in sus_tlds)),
    ]
    return np.array(feats, dtype=np.float32)

# ─────────────────────────────────────────────────────────
# HELPER: LAYER 2 PREDICTION
# ─────────────────────────────────────────────────────────
def run_layer2(url: str) -> dict:
    """
    Fetch HTML, extract features, run Layer 2 model.
    Returns score and prediction.
    """
    # 1. Fetch HTML
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        resp = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
        raw_html = resp.text
        # resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        # raw_html = resp.text
    except Exception as e:
        logger.warning(f"Layer 2 HTML fetch failed: {e}")
        return {"score": None, "prediction": None, "error": str(e)}

    # 2. Clean HTML → text
    clean_text = clean_html_to_text(raw_html)
    logger.info(f"HTML length: {len(raw_html)} | Clean text: {len(clean_text)} chars")
    logger.info(f"Clean text preview: {clean_text[:200]}")
    url_dbg = extract_url_features(url)
    logger.info(f"URL features: {url_dbg}")
    if len(clean_text) < 30:
        logger.warning("Layer 2: cleaned HTML too short")
        return {"score": None, "prediction": None, "error": "HTML content too short"}

    # 3. HTML → TF-IDF → SVD → dense
    hv_matrix  = hv.transform([clean_text])                          # sparse
    tfidf_mat  = l2_tfidf.transform(hv_matrix).astype(np.float32)   # sparse
    html_dense = l2_svd.transform(tfidf_mat).astype(np.float32)      # (1, 256)

    # 4. URL heuristic features → scale
    url_feats        = extract_url_features(url).reshape(1, -1)
    url_feats_scaled = l2_scaler.transform(url_feats).astype(np.float32)

    # 5. Predict
    score = float(l2_model.predict(
        [html_dense, url_feats_scaled], verbose=0
    )[0][0])

    prediction = "phishing" if score >= l2_threshold else "legitimate"
    return {"score": round(score, 4), "prediction": prediction, "error": None}

# ─────────────────────────────────────────────────────────
# COMBINATION LOGIC
# ─────────────────────────────────────────────────────────
def combine_results(l1: Optional[dict], l2: dict) -> dict:
    """
    Combine Layer 1 and Layer 2 scores into final decision.

    Rules:
      - L2 phishing                          → PHISHING  (L2 has lower FNR)
      - L1 phishing + L2 legitimate          → SUSPICIOUS
      - Both legitimate                      → LEGITIMATE
      - HTTP (no L1) + L2 legitimate         → LEGITIMATE (with HTTP warning)
      - HTTP (no L1) + L2 phishing           → PHISHING
    """
    l2_score      = l2.get("score")
    l2_prediction = l2.get("prediction")

    if l2_score is None:
        # Layer 2 failed — fall back to Layer 1 only if available
        if l1 and l1.get("score") is not None:
            l1_pred = l1["prediction"]
            return {
                "verdict":     l1_pred,
                "confidence":  "low",
                "reason":      "Layer 2 unavailable — Layer 1 only",
                "http_warning": False,
            }
        return {
            "verdict":     "unknown",
            "confidence":  "none",
            "reason":      "Both layers failed",
            "http_warning": False,
        }

    is_http = l1 is None  # no Layer 1 means HTTP

    if l1 is not None:
        l1_score      = l1.get("score", 0)
        l1_prediction = l1.get("prediction", "legitimate")

        if l2_prediction == "phishing" and l1_prediction == "phishing":
            verdict    = "phishing"
            confidence = "very_high"
            reason     = f"Both layers agree — L1={l1_score:.3f}, L2={l2_score:.3f}"

        elif l2_prediction == "phishing" and l1_prediction == "legitimate":
            verdict    = "phishing"
            confidence = "high"
            reason     = f"Layer 2 detected phishing in HTML content — L2={l2_score:.3f}"

        elif l2_prediction == "legitimate" and l1_prediction == "phishing":
            verdict    = "suspicious"
            confidence = "medium"
            reason     = f"Layer 1 flagged TLS anomalies but HTML appears clean — L1={l1_score:.3f}, L2={l2_score:.3f}"

        else:  # both legitimate
            verdict    = "legitimate"
            confidence = "high"
            reason     = f"Both layers agree — L1={l1_score:.3f}, L2={l2_score:.3f}"

        return {
            "verdict":      verdict,
            "confidence":   confidence,
            "reason":       reason,
            "http_warning": False,
        }

    else:
        # HTTP — Layer 2 only
        if l2_prediction == "phishing":
            return {
                "verdict":      "phishing",
                "confidence":   "medium",
                "reason":       f"Layer 2 detected phishing — L2={l2_score:.3f} (HTTP site, no TLS analysis)",
                "http_warning": True,
            }
        else:
            return {
                "verdict":      "legitimate",
                "confidence":   "low",
                "reason":       f"Layer 2 found no phishing signals — L2={l2_score:.3f}",
                "http_warning": True,
            }

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Two-Layer Phishing Detection API",
        "layer1_threshold": l1_threshold,
        "layer2_threshold": l2_threshold,
    }

@app.post("/predict")
def predict_url(payload: URLInput):
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    parsed = urlparse(url)
    is_https = parsed.scheme == "https"
    is_http  = parsed.scheme == "http"

    if not is_https and not is_http:
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    logger.info(f"Predicting: {url}  (scheme={parsed.scheme})")

    # ── Layer 1: HTTPS only ───────────────────────────
    l1_result = None
    if is_https:
        try:
            xml_path = capturer.capture_single_url(url)
            if xml_path is None:
                logger.warning("Layer 1 TLS capture failed — skipping")
            else:
                df_feat = extract_features_from_url_and_xml(url, xml_path)
                # Keep only the features used during training
                df_feat = df_feat[l1_features]
                X_scaled = l1_scaler.transform(df_feat.values)
                l1_score = float(l1_model.predict(X_scaled, verbose=0)[0][0])
                l1_pred  = "phishing" if l1_score >= l1_threshold else "legitimate"
                l1_result = {"score": round(l1_score, 4), "prediction": l1_pred}
                logger.info(f"Layer 1 → score={l1_score:.4f}, pred={l1_pred}")
        except Exception as e:
            logger.error(f"Layer 1 error: {e}")
            l1_result = None

    # ── Layer 2: always runs ──────────────────────────
    l2_result = run_layer2(url)
    logger.info(f"Layer 2 → score={l2_result.get('score')}, pred={l2_result.get('prediction')}")

    # ── Combine ───────────────────────────────────────
    final = combine_results(l1_result, l2_result)

    return {
        "url":        url,
        "scheme":     parsed.scheme,
        "verdict":    final["verdict"],        # phishing / legitimate / suspicious / unknown
        "confidence": final["confidence"],     # very_high / high / medium / low / none
        "reason":     final["reason"],
        "http_warning": final["http_warning"], # True if HTTP site (no encryption)
        "layer1": {
            "ran":        l1_result is not None,
            "score":      l1_result["score"]      if l1_result else None,
            "prediction": l1_result["prediction"] if l1_result else "skipped",
            "threshold":  l1_threshold,
        },
        "layer2": {
            "ran":        l2_result.get("score") is not None,
            "score":      l2_result.get("score"),
            "prediction": l2_result.get("prediction"),
            "threshold":  l2_threshold,
            "error":      l2_result.get("error"),
        },
    }


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__mainn__":
    import uvicorn
    uvicorn.run("mainn:app", host="0.0.0.0", port=8000, reload=True)