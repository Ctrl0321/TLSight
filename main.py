import re
import io
import csv
import asyncio
import pickle
import logging
from typing import Optional, List
from urllib.parse import urlparse, urljoin
import json
import pandas as pd

import numpy as np
import requests
import tensorflow as tf
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tls_capture import TLSTrafficCapture
from feature_extractor import extract_features_from_url_and_xml


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
MAX_TEXT_LEN    = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# LOAD ALL MODELS AT STARTUP
# ─────────────────────────────────────────────────────────
logger.info("Loading Layer 1 artifacts...")
l1_model     = tf.keras.models.load_model(LAYER1_MODEL_PATH, compile=False)
l1_scaler    = pickle.load(open(LAYER1_SCALER_PATH,   "rb"))
l1_features  = pickle.load(open(LAYER1_FEATURES_PATH, "rb"))
l1_threshold = pickle.load(open(LAYER1_THRESHOLD_PATH,"rb"))
logger.info("Layer 1 loaded — threshold=%s, features=%d", l1_threshold, len(l1_features))

logger.info("Loading Layer 2 artifacts...")
l2_model      = tf.keras.models.load_model(LAYER2_MODEL_PATH, compile=False)
l2_tfidf      = pickle.load(open(LAYER2_TFIDF_PATH,      "rb"))
l2_svd        = pickle.load(open(LAYER2_SVD_PATH,        "rb"))
l2_dom_scaler = pickle.load(open(LAYER2_DOM_SCALER_PATH, "rb"))
l2_threshold  = pickle.load(open(LAYER2_THRESHOLD_PATH,  "rb"))
l2_vocab_meta = pickle.load(open(LAYER2_CHAR_VOCAB_PATH, "rb"))

CHAR2IDX         = l2_vocab_meta["CHAR2IDX"]
UNK_IDX          = l2_vocab_meta["UNK_IDX"]
MAX_URL_LEN      = l2_vocab_meta["MAX_URL_LEN"]
DOM_FEATURE_NAMES = l2_vocab_meta.get("DOM_FEATURE_NAMES", [])

logger.info(
    "Layer 2 loaded — threshold=%s, svd_dim=%s, max_url_len=%s",
    l2_threshold, getattr(l2_svd, "n_components", "unknown"), MAX_URL_LEN,
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

# ─────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────
class URLInput(BaseModel):
    url: str

class BulkURLInput(BaseModel):
    urls: List[str]

# ─────────────────────────────────────────────────────────
# URL RESOLUTION HELPERS
# ─────────────────────────────────────────────────────────

VALID_SCHEMES = {"http", "https"}

# Matches anything that LOOKS like a scheme prefix (letters + "://")
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+\-.]*):\/\/", re.IGNORECASE)


def parse_raw_url(raw: str) -> tuple[str, str]:
    """
    Accepts any raw string the user typed.

    Returns (resolved_url, effective_scheme) where resolved_url has a
    valid http/https scheme and effective_scheme is 'https' or 'http'.

    Raises ValueError with a human-readable message for invalid inputs.

    Rules
    ─────
    1. Strip whitespace.
    2. Empty string → ValueError.
    3. If a scheme-like prefix is present (letters + "://"):
         • If it's http or https → keep as-is.
         • Otherwise             → raise "Invalid URL scheme".
    4. No scheme present:
         • Try https://  first  (HEAD request, 6 s timeout).
         • If that reaches the server → return https://...
         • Otherwise try http://      → return http://...
         • If both fail              → raise "URL unreachable".
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("URL cannot be empty.")

    m = _SCHEME_RE.match(raw)
    if m:
        scheme = m.group(1).lower()
        if scheme not in VALID_SCHEMES:
            raise ValueError(
                f"Invalid URL scheme '{scheme}'. Only http and https are supported."
            )
        # User gave a valid scheme; return it directly — no probe needed.
        return raw, scheme

    # ── No scheme: probe https first, then http ───────────────────────
    for scheme in ("https", "http"):
        candidate = f"{scheme}://{raw}"
        try:
            resp = requests.head(
                candidate,
                timeout=6,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 PhishingDetector/1.0"},
            )
            # Any HTTP response (even 4xx/5xx) means the host is reachable
            effective_scheme = urlparse(resp.url).scheme.lower()
            if effective_scheme not in VALID_SCHEMES:
                effective_scheme = scheme
            return resp.url if resp.url else candidate, effective_scheme
        except requests.RequestException:
            continue

    raise ValueError(
        f"URL '{raw}' is unreachable over both HTTPS and HTTP. "
        "Please check the address and try again."
    )


def normalize_url_for_page(url: str) -> str:
    """Used for DOM joining / absolute URL resolution. Keeps full scheme."""
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    return url


def normalize_url_for_model(url: str) -> str:
    """
    Make live URLs closer to training CSV format.
    Removes scheme, keeps host + path + query, lowercase.
    """
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    p   = urlparse(url)
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
    parts = [soup.get_text(" ", strip=True)]
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
    return link in {"", "#", "#content", "javascript:void(0);",
                    "javascript:void(0)", "javascript:;", "about:blank"}


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
    soup      = BeautifulSoup(raw_html, "lxml")
    page_url  = normalize_url_for_page(page_url)
    base_host = (urlparse(page_url).hostname or "").lower()

    total_links = script_files = css_files = img_files = 0
    a_files = a_null = null_links = internal = external = error_links = 0
    all_links: list[str] = []

    for tag in soup.find_all("script", src=True):
        script_files += 1; total_links += 1
        all_links.append(urljoin(page_url, tag.get("src", "")))

    for tag in soup.find_all("link", href=True):
        css_files += 1; total_links += 1
        all_links.append(urljoin(page_url, tag.get("href", "")))

    for tag in soup.find_all("img", src=True):
        img_files += 1; total_links += 1
        all_links.append(urljoin(page_url, tag.get("src", "")))

    for tag in soup.find_all("a"):
        a_files += 1; total_links += 1
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
        action     = form.get("action", "")
        action_abs = urljoin(page_url, action)
        if is_null_link(action) or not is_valid_http_url(action_abs) or not same_domain(base_host, action_abs):
            suspicious_forms += 1

    return np.array([
        safe_div(script_files, total_links),
        safe_div(css_files,    total_links),
        safe_div(img_files,    total_links),
        safe_div(a_files,      total_links),
        safe_div(a_null,       total_links),
        safe_div(null_links,   total_links),
        float(total_links),
        safe_div(internal,     total_links),
        safe_div(external,     total_links),
        safe_div(external,     internal),
        safe_div(error_links,  total_links),
        float(total_forms),
        safe_div(suspicious_forms, total_forms),
    ], dtype=np.float32)

# ─────────────────────────────────────────────────────────
# REQUEST / FETCH
# ─────────────────────────────────────────────────────────

def fetch_html(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection":      "keep-alive",
    }
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)
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
        return {"score": None, "prediction": None, "error": str(e)}

    text_blob = extract_text_and_noisy_html(raw_html)
    if len(text_blob) < 20:
        return {"score": None, "prediction": None, "error": "HTML content too short"}

    page_url  = normalize_url_for_page(final_url or url)
    model_url = normalize_url_for_model(final_url or url)

    url_char_seq     = encode_url_chars(model_url).reshape(1, -1)
    tfidf_mat        = l2_tfidf.transform([text_blob]).astype(np.float32)
    html_dense       = l2_svd.transform(tfidf_mat).astype(np.float32)
    dom_feats        = extract_dom_features(raw_html, page_url).reshape(1, -1)
    dom_feats_scaled = l2_dom_scaler.transform(dom_feats).astype(np.float32)

    score = float(
        l2_model.predict(
            {"url_char_input": url_char_seq, "html_text_input": html_dense, "dom_input": dom_feats_scaled},
            verbose=0,
        )[0][0]
    )

    return {
        "score":       round(score, 4),
        "prediction":  "phishing" if score >= l2_threshold else "legitimate",
        "error":       None,
        "final_url":   page_url,
        "model_url":   model_url,
        "dom_features": {
            name: float(val)
            for name, val in zip(DOM_FEATURE_NAMES, dom_feats.flatten().tolist())
        } if DOM_FEATURE_NAMES else None,
    }

# ─────────────────────────────────────────────────────────
# COMBINATION LOGIC
# ─────────────────────────────────────────────────────────

def combine_results(l1: Optional[dict], l2: dict, l1_skip_reason: Optional[str] = None) -> dict:
    l2_score      = l2.get("score")
    l2_prediction = l2.get("prediction")

    if l2_score is None:
        if l1 and l1.get("score") is not None:
            return {"verdict": l1["prediction"], "confidence": "low",
                    "reason": "Layer 2 unavailable — Layer 1 only", "http_warning": False}
        return {"verdict": "unknown", "confidence": "none",
                "reason": "Both layers failed", "http_warning": False}

    if l1 is not None:
        l1_score      = l1.get("score", 0.0)
        l1_prediction = l1.get("prediction", "legitimate")

        if l2_prediction == "phishing" and l1_prediction == "phishing":
            verdict, confidence = "phishing",  "very_high"
            reason = f"Both layers agree : L1={l1_score:.3f}, L2={l2_score:.3f}"
        elif l2_prediction == "phishing" and l1_prediction == "legitimate":
            verdict, confidence = "phishing",  "high"
            reason = f"Layer 2 detected phishing content/DOM patterns — L2={l2_score:.3f}"
        elif l2_prediction == "legitimate" and l1_prediction == "phishing":
            verdict, confidence = "suspicious", "medium"
            reason = f"Layer 1 flagged TLS anomalies but Layer 2 looks clean — L1={l1_score:.3f}, L2={l2_score:.3f}"
        else:
            verdict, confidence = "legitimate", "high"
            reason = f"Both layers agree — L1={l1_score:.3f}, L2={l2_score:.3f}"

        return {"verdict": verdict, "confidence": confidence,
                "reason": reason, "http_warning": False}

    # Layer 1 was skipped
    http_warning = l1_skip_reason is not None and "http" in (l1_skip_reason or "").lower()

    if l2_prediction == "phishing":
        return {
            "verdict":      "phishing",
            "confidence":   "medium",
            "reason":       f"Layer 2 detected phishing — L2={l2_score:.3f} (Layer 1 skipped: {l1_skip_reason or 'N/A'})",
            "http_warning": http_warning,
        }

    return {
        "verdict":      "legitimate",
        "confidence":   "low",
        "reason":       f"Layer 2 found no phishing signals — L2={l2_score:.3f} (Layer 1 skipped: {l1_skip_reason or 'N/A'})",
        "http_warning": http_warning,
    }

# ─────────────────────────────────────────────────────────
# CORE PREDICTION LOGIC (shared by single + bulk)
# ─────────────────────────────────────────────────────────

def predict_single(raw_url: str) -> dict:
    """
    Returns a full prediction dict for one raw URL string.
    Never raises — errors are embedded in the response.
    """
    raw_url = raw_url.strip()
    if not raw_url:
        return _error_response(raw_url, "URL cannot be empty.")

    # ── Resolve URL and determine scheme ────────────────
    try:
        resolved_url, effective_scheme = parse_raw_url(raw_url)
    except ValueError as e:
        return _error_response(raw_url, str(e))

    parsed      = urlparse(resolved_url)
    is_https    = effective_scheme == "https"
    is_http     = effective_scheme == "http"

    logger.info("Predicting raw=%s resolved=%s scheme=%s", raw_url, resolved_url, effective_scheme)

    # ── Layer 1: HTTPS only ──────────────────────────────
    l1_result       = None
    l1_skip_reason  = None

    if is_https:
        try:
            xml_path = capturer.capture_single_url(resolved_url)
            if xml_path is None:
                l1_skip_reason = "TLS capture returned no data"
                logger.warning("Layer 1 TLS capture failed — skipping")
            else:
                df_feat  = extract_features_from_url_and_xml(resolved_url, xml_path)
                df_feat  = df_feat[l1_features]
                X_scaled = l1_scaler.transform(df_feat.values)
                l1_score = float(l1_model.predict(X_scaled, verbose=0)[0][0])
                l1_pred  = "phishing" if l1_score >= l1_threshold else "legitimate"
                l1_result = {"score": round(l1_score, 4), "prediction": l1_pred}
                logger.info("Layer 1 → score=%.4f, pred=%s", l1_score, l1_pred)
        except Exception as e:
            l1_skip_reason = f"Layer 1 error: {e}"
            logger.error("Layer 1 error: %s", e)
    elif is_http:
        l1_skip_reason = "Site is only reachable over HTTP — TLS analysis not possible"

    # ── Layer 2: always runs ────────────────────────────
    l2_result = run_layer2(resolved_url)
    logger.info("Layer 2 → score=%s, pred=%s", l2_result.get("score"), l2_result.get("prediction"))

    final = combine_results(l1_result, l2_result, l1_skip_reason)

    return {
        "input_url":    raw_url,
        "url":          resolved_url,
        "scheme":       effective_scheme,
        "verdict":      final["verdict"],
        "confidence":   final["confidence"],
        "reason":       final["reason"],
        "http_warning": final["http_warning"],
        "layer1": {
            "ran":        l1_result is not None,
            "skipped":    l1_result is None,
            "skip_reason": l1_skip_reason,
            "score":      l1_result["score"]      if l1_result else None,
            "prediction": l1_result["prediction"] if l1_result else "skipped",
            "threshold":  l1_threshold,
        },
        "layer2": {
            "ran":         l2_result.get("score") is not None,
            "score":       l2_result.get("score"),
            "prediction":  l2_result.get("prediction"),
            "threshold":   l2_threshold,
            "error":       l2_result.get("error"),
            "final_url":   l2_result.get("final_url"),
            "model_url":   l2_result.get("model_url"),
            "dom_features": l2_result.get("dom_features"),
        },
        "error": None,
    }


def _error_response(raw_url: str, message: str) -> dict:
    return {
        "input_url":    raw_url,
        "url":          raw_url,
        "scheme":       None,
        "verdict":      "invalid",
        "confidence":   "none",
        "reason":       message,
        "http_warning": False,
        "layer1":       {"ran": False, "skipped": True, "skip_reason": message,
                         "score": None, "prediction": "skipped", "threshold": l1_threshold},
        "layer2":       {"ran": False, "score": None, "prediction": None,
                         "threshold": l2_threshold, "error": message,
                         "final_url": None, "model_url": None, "dom_features": None},
        "error": message,
    }

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status":            "ok",
        "message":           "Two-Layer Phishing Detection API",
        "layer1_threshold":  l1_threshold,
        "layer2_threshold":  l2_threshold,
    }


@app.post("/predict")
def predict_url(payload: URLInput):
    raw_url = payload.url.strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    return predict_single(raw_url)


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    import json
    import pandas as pd
    import io

    # ─────────────────────────────────────────
    # READ FILE SAFELY
    # ─────────────────────────────────────────
    content = await file.read()

    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ─────────────────────────────────────────
    # DECODE (MULTI-ENCODING SUPPORT)
    # ─────────────────────────────────────────
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

    if not text.strip():
        raise HTTPException(status_code=400, detail="CSV file has no readable content.")

 
    try:
        df = pd.read_csv(
            io.StringIO(text),
            engine="python",          # tolerant parser
            on_bad_lines="skip",      # skip broken rows
            skip_blank_lines=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV parsing failed: {str(e)}"
        )

    # ─────────────────────────────────────────
    # VALIDATE DATAFRAME
    # ─────────────────────────────────────────
    if df.empty or df.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="No columns found in CSV. Make sure it has at least one column (e.g. 'url')."
        )

    fieldnames = df.columns.tolist()

    # ─────────────────────────────────────────
    # DETECT URL COLUMN
    # ─────────────────────────────────────────
    url_col = None
    for col in fieldnames:
        if col.strip().lower() in {"url", "urls", "link", "links", "domain", "website"}:
            url_col = col
            break
    
    print("DF HEAD:\n", df.head())
    print("COLUMNS:", df.columns.tolist())
    print("ROWS COUNT:", len(df))

    if url_col is None:
        url_col = fieldnames[0]  # fallback
    rows = (
        df[url_col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .tolist()
    )

    rows = [r for r in rows if r]

    if not rows:
        raise HTTPException(status_code=400, detail="No valid URLs found in CSV.")

    if len(rows) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 URLs per CSV upload.")

    # ─────────────────────────────────────────
    # STREAM RESPONSE
    # ─────────────────────────────────────────
    def stream():
        # Send metadata first
        yield json.dumps({"type": "meta", "total": len(rows)}) + "\n"

        for idx, raw in enumerate(rows):
            try:
                result = predict_single(raw)
            except Exception as e:
                result = {
                    "input_url": raw,
                    "url": raw,
                    "verdict": "unknown",
                    "confidence": "none",
                    "reason": f"Processing error: {str(e)}",
                    "error": str(e),
                }

            result["_index"] = idx
            result["type"] = "result"

            yield json.dumps(result) + "\n"

        yield json.dumps({"type": "done", "total": len(rows)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)