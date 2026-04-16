#!/usr/bin/env python3
# ============================================================
# TWO-LAYER PHISHING DETECTION API
# Layer 1: TLS/URL tabular features  (HTTPS only)
# Layer 2: Paper-style NN
#   - URL character sequence branch
#   - HTML text TF-IDF + SVD branch
#   - DOM structured features branch
# ============================================================

import re
import pickle
import logging
from typing import Optional
from urllib.parse import urlparse, urljoin

import numpy as np
import requests
import tensorflow as tf
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tls_capture import TLSTrafficCapture
from feature_extractor import extract_features_from_url_and_xml

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
LAYER1_MODEL_PATH      = "Model/Layer-1/layer1_model.keras"
LAYER1_SCALER_PATH     = "Model/Layer-1/layer1_scaler.pkl"
LAYER1_FEATURES_PATH   = "Model/Layer-1/layer1_features.pkl"
LAYER1_THRESHOLD_PATH  = "Model/Layer-1/layer1_threshold.pkl"

LAYER2_MODEL_PATH      = "Model/Layer-2/layer2_paper_nn.keras"
LAYER2_TFIDF_PATH      = "Model/Layer-2/layer2_text_tfidf.pkl"
LAYER2_SVD_PATH        = "Model/Layer-2/layer2_text_svd.pkl"
LAYER2_DOM_SCALER_PATH = "Model/Layer-2/layer2_dom_scaler.pkl"
LAYER2_CHAR_VOCAB_PATH = "Model/Layer-2/layer2_char_vocab.pkl"
LAYER2_THRESHOLD_PATH  = "Model/Layer-2/layer2_threshold.pkl"

REQUEST_TIMEOUT = 12
MAX_TEXT_LEN = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# LOAD ALL MODELS AT STARTUP
# ─────────────────────────────────────────────────────────
logger.info("Loading Layer 1 artifacts...")
l1_model = tf.keras.models.load_model(LAYER1_MODEL_PATH, compile=False)
l1_scaler = pickle.load(open(LAYER1_SCALER_PATH, "rb"))
l1_features = pickle.load(open(LAYER1_FEATURES_PATH, "rb"))
l1_threshold = pickle.load(open(LAYER1_THRESHOLD_PATH, "rb"))
logger.info("Layer 1 loaded — threshold=%s, features=%d", l1_threshold, len(l1_features))

logger.info("Loading Layer 2 artifacts...")
l2_model = tf.keras.models.load_model(LAYER2_MODEL_PATH, compile=False)
l2_tfidf = pickle.load(open(LAYER2_TFIDF_PATH, "rb"))
l2_svd = pickle.load(open(LAYER2_SVD_PATH, "rb"))
l2_dom_scaler = pickle.load(open(LAYER2_DOM_SCALER_PATH, "rb"))
l2_threshold = pickle.load(open(LAYER2_THRESHOLD_PATH, "rb"))
l2_vocab_meta = pickle.load(open(LAYER2_CHAR_VOCAB_PATH, "rb"))

CHAR2IDX = l2_vocab_meta["CHAR2IDX"]
UNK_IDX = l2_vocab_meta["UNK_IDX"]
MAX_URL_LEN = l2_vocab_meta["MAX_URL_LEN"]
DOM_FEATURE_NAMES = l2_vocab_meta.get("DOM_FEATURE_NAMES", [])

logger.info(
    "Layer 2 loaded — threshold=%s, svd_dim=%s, max_url_len=%s",
    l2_threshold,
    getattr(l2_svd, "n_components", "unknown"),
    MAX_URL_LEN,
)

capturer = TLSTrafficCapture()

# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(title="Two-Layer Phishing Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

# ─────────────────────────────────────────────────────────
# URL / INPUT HELPERS
# ─────────────────────────────────────────────────────────
def ensure_scheme(url: str) -> str:
    """
    Accepts:
      cricbuzz.com
      www.cricbuzz.com
      https://cricbuzz.com
      http://cricbuzz.com
    Returns a valid URL with scheme.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    return url

def normalize_url_for_page(url: str) -> str:
    """
    Used for DOM joining / absolute URL resolution.
    Keeps full scheme.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    return url

def normalize_url_for_model(url: str) -> str:
    """
    Make live URLs closer to training CSV format.
    Removes scheme, keeps host + path + query, lowercase.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""

    if "://" not in url:
        url = "https://" + url

    p = urlparse(url)
    out = (p.netloc + p.path + (f"?{p.query}" if p.query else "")).lower().strip()

    if out.endswith("/") and len(out) > 1:
        out = out[:-1]

    return out

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0

# ─────────────────────────────────────────────────────────
# LAYER 2 HELPERS — MUST MATCH TRAINING
# ─────────────────────────────────────────────────────────
def encode_url_chars(url: str) -> np.ndarray:
    if not isinstance(url, str):
        url = ""
    url = url.lower().strip()
    seq = [CHAR2IDX.get(ch, UNK_IDX) for ch in url[:MAX_URL_LEN]]
    if len(seq) < MAX_URL_LEN:
        seq += [0] * (MAX_URL_LEN - len(seq))
    return np.array(seq, dtype=np.int32)

def extract_text_and_noisy_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")

    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    parts = []

    visible_text = soup.get_text(" ", strip=True)
    if visible_text:
        parts.append(visible_text)

    for tag_name in ["title", "meta", "body", "form", "div", "h1", "h2"]:
        for tag in soup.find_all(tag_name):
            if tag_name == "meta":
                for attr in ["name", "content", "property", "http-equiv"]:
                    v = tag.get(attr)
                    if v:
                        parts.append(str(v))
            else:
                for attr_name, attr_val in tag.attrs.items():
                    if isinstance(attr_val, list):
                        attr_val = " ".join(map(str, attr_val))
                    parts.append(f"{attr_name} {attr_val}")

    merged = " ".join(parts).lower()
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged[:MAX_TEXT_LEN]

def is_null_link(link: str) -> bool:
    if not link:
        return True
    link = link.strip().lower()
    return link in {
        "",
        "#",
        "#content",
        "javascript:void(0);",
        "javascript:void(0)",
        "javascript:;",
        "about:blank",
    }

def is_valid_http_url(link: str) -> bool:
    try:
        p = urlparse(link)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False

def same_domain(base_host: str, link: str) -> bool:
    try:
        host = (urlparse(link).hostname or "").lower()
        return host == base_host
    except Exception:
        return False

def extract_dom_features(raw_html: str, page_url: str) -> np.ndarray:
    """
    Must match training logic exactly.
    Returns 13 features.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    page_url = normalize_url_for_page(page_url)
    base_host = (urlparse(page_url).hostname or "").lower()

    total_links = 0
    script_files = 0
    css_files = 0
    img_files = 0
    a_files = 0
    a_null = 0
    null_links = 0
    internal = 0
    external = 0
    error_links = 0

    all_links = []

    for tag in soup.find_all("script", src=True):
        script_files += 1
        total_links += 1
        src = tag.get("src", "")
        all_links.append(urljoin(page_url, src))

    for tag in soup.find_all("link", href=True):
        css_files += 1
        total_links += 1
        href = tag.get("href", "")
        all_links.append(urljoin(page_url, href))

    for tag in soup.find_all("img", src=True):
        img_files += 1
        total_links += 1
        src = tag.get("src", "")
        all_links.append(urljoin(page_url, src))

    for tag in soup.find_all("a"):
        a_files += 1
        total_links += 1
        href = tag.get("href", "")
        if not href:
            a_null += 1
        if is_null_link(href):
            null_links += 1
        else:
            all_links.append(urljoin(page_url, href))

    for link in all_links:
        if not is_valid_http_url(link):
            error_links += 1
            continue
        if same_domain(base_host, link):
            internal += 1
        else:
            external += 1

    forms = soup.find_all("form")
    total_forms = len(forms)
    suspicious_forms = 0

    for form in forms:
        action = form.get("action", "")
        if is_null_link(action):
            suspicious_forms += 1
            continue

        action_abs = urljoin(page_url, action)

        if not is_valid_http_url(action_abs):
            suspicious_forms += 1
            continue

        if not same_domain(base_host, action_abs):
            suspicious_forms += 1

    feat = np.array([
        safe_div(script_files, total_links),     # F3
        safe_div(css_files, total_links),        # F4
        safe_div(img_files, total_links),        # F5
        safe_div(a_files, total_links),          # F6
        safe_div(a_null, total_links),           # F7
        safe_div(null_links, total_links),       # F8
        float(total_links),                      # F9
        safe_div(internal, total_links),         # F10
        safe_div(external, total_links),         # F11
        safe_div(external, internal),            # F12
        safe_div(error_links, total_links),      # F13
        float(total_forms),                      # F14
        safe_div(suspicious_forms, total_forms)  # F15
    ], dtype=np.float32)

    return feat

# ─────────────────────────────────────────────────────────
# REQUEST / FETCH HELPERS
# ─────────────────────────────────────────────────────────
def fetch_html(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    resp = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers=headers,
        allow_redirects=True
    )
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type}")

    return resp.text, str(resp.url), content_type

# ─────────────────────────────────────────────────────────
# LAYER 2 PREDICTION
# ─────────────────────────────────────────────────────────
def run_layer2(url: str) -> dict:
    try:
        raw_html, final_url, content_type = fetch_html(url)
    except Exception as e:
        logger.warning("Layer 2 HTML fetch failed: %s", e)
        return {
            "score": None,
            "prediction": None,
            "error": str(e),
        }

    text_blob = extract_text_and_noisy_html(raw_html)
    if len(text_blob) < 20:
        logger.warning("Layer 2: extracted HTML text too short")
        return {
            "score": None,
            "prediction": None,
            "error": "HTML content too short",
        }

    page_url = normalize_url_for_page(final_url or url)
    model_url = normalize_url_for_model(final_url or url)

    url_char_seq = encode_url_chars(model_url).reshape(1, -1)

    tfidf_mat = l2_tfidf.transform([text_blob]).astype(np.float32)
    html_dense = l2_svd.transform(tfidf_mat).astype(np.float32)

    dom_feats = extract_dom_features(raw_html, page_url).reshape(1, -1)
    dom_feats_scaled = l2_dom_scaler.transform(dom_feats).astype(np.float32)

    logger.info("Layer 2 final_url=%s", page_url)
    logger.info("Layer 2 model_url=%s", model_url)
    logger.info("Layer 2 content_type=%s", content_type)
    logger.info("Layer 2 raw_html_len=%d text_blob_len=%d", len(raw_html), len(text_blob))
    logger.info("Layer 2 dom_feats=%s", dom_feats.tolist())

    # IMPORTANT: named inputs remove ambiguity
    score = float(
        l2_model.predict(
            {
                "url_char_input": url_char_seq,
                "html_text_input": html_dense,
                "dom_input": dom_feats_scaled,
            },
            verbose=0
        )[0][0]
    )

    prediction = "phishing" if score >= l2_threshold else "legitimate"

    return {
        "score": round(score, 4),
        "prediction": prediction,
        "error": None,
        "final_url": page_url,
        "model_url": model_url,
        "dom_features": {
            name: float(val)
            for name, val in zip(DOM_FEATURE_NAMES, dom_feats.flatten().tolist())
        } if DOM_FEATURE_NAMES else None,
    }

# ─────────────────────────────────────────────────────────
# COMBINATION LOGIC
# ─────────────────────────────────────────────────────────
def combine_results(l1: Optional[dict], l2: dict) -> dict:
    l2_score = l2.get("score")
    l2_prediction = l2.get("prediction")

    if l2_score is None:
        if l1 and l1.get("score") is not None:
            return {
                "verdict": l1["prediction"],
                "confidence": "low",
                "reason": "Layer 2 unavailable — Layer 1 only",
                "http_warning": False,
            }
        return {
            "verdict": "unknown",
            "confidence": "none",
            "reason": "Both layers failed",
            "http_warning": False,
        }

    if l1 is not None:
        l1_score = l1.get("score", 0.0)
        l1_prediction = l1.get("prediction", "legitimate")

        if l2_prediction == "phishing" and l1_prediction == "phishing":
            verdict = "phishing"
            confidence = "very_high"
            reason = f"Both layers agree : L1={l1_score:.3f}, L2={l2_score:.3f}"

        elif l2_prediction == "phishing" and l1_prediction == "legitimate":
            verdict = "phishing"
            confidence = "high"
            reason = f"Layer 2 detected phishing content/DOM patterns — L2={l2_score:.3f}"

        elif l2_prediction == "legitimate" and l1_prediction == "phishing":
            verdict = "suspicious"
            confidence = "medium"
            reason = f"Layer 1 flagged TLS anomalies but Layer 2 looks clean — L1={l1_score:.3f}, L2={l2_score:.3f}"

        else:
            verdict = "legitimate"
            confidence = "high"
            reason = f"Both layers agree — L1={l1_score:.3f}, L2={l2_score:.3f}"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "http_warning": False,
        }

    if l2_prediction == "phishing":
        return {
            "verdict": "phishing",
            "confidence": "medium",
            "reason": f"Layer 2 detected phishing — L2={l2_score:.3f} (HTTP site, no TLS analysis)",
            "http_warning": True,
        }

    return {
        "verdict": "legitimate",
        "confidence": "low",
        "reason": f"Layer 2 found no phishing signals — L2={l2_score:.3f}",
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
    raw_url = payload.url.strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    url = ensure_scheme(raw_url)
    parsed = urlparse(url)

    is_https = parsed.scheme == "https"
    is_http = parsed.scheme == "http"

    if not is_https and not is_http:
        raise HTTPException(status_code=400, detail="URL must be a valid http(s) URL")

    logger.info("Predicting raw=%s normalized=%s scheme=%s", raw_url, url, parsed.scheme)

    # ── Layer 1: HTTPS only ───────────────────────────
    l1_result = None
    if is_https:
        try:
            xml_path = capturer.capture_single_url(url)
            if xml_path is None:
                logger.warning("Layer 1 TLS capture failed — skipping")
            else:
                df_feat = extract_features_from_url_and_xml(url, xml_path)
                df_feat = df_feat[l1_features]
                X_scaled = l1_scaler.transform(df_feat.values)
                l1_score = float(l1_model.predict(X_scaled, verbose=0)[0][0])
                l1_pred = "phishing" if l1_score >= l1_threshold else "legitimate"
                l1_result = {
                    "score": round(l1_score, 4),
                    "prediction": l1_pred,
                }
                logger.info("Layer 1 → score=%.4f, pred=%s", l1_score, l1_pred)
        except Exception as e:
            logger.error("Layer 1 error: %s", e)
            l1_result = None

    # ── Layer 2: always runs ──────────────────────────
    l2_result = run_layer2(url)
    logger.info("Layer 2 → score=%s, pred=%s", l2_result.get("score"), l2_result.get("prediction"))

    final = combine_results(l1_result, l2_result)

    return {
        "input_url": raw_url,
        "url": url,
        "scheme": parsed.scheme,
        "verdict": final["verdict"],
        "confidence": final["confidence"],
        "reason": final["reason"],
        "http_warning": final["http_warning"],
        "layer1": {
            "ran": l1_result is not None,
            "score": l1_result["score"] if l1_result else None,
            "prediction": l1_result["prediction"] if l1_result else "skipped",
            "threshold": l1_threshold,
        },
        "layer2": {
            "ran": l2_result.get("score") is not None,
            "score": l2_result.get("score"),
            "prediction": l2_result.get("prediction"),
            "threshold": l2_threshold,
            "error": l2_result.get("error"),
            "final_url": l2_result.get("final_url"),
            "model_url": l2_result.get("model_url"),
            "dom_features": l2_result.get("dom_features"),
        },
    }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)