"""
STEP 6 — Bridge API (Local FastAPI Server)
==========================================
This runs on YOUR PC (hospital machine).
It acts as a bridge between the React dashboard and the Render cloud.

Flow:
  Dashboard → Bridge API (localhost:5000)
           → Render Cloud (encrypted query)
           → Decrypt result using secret key
           → Return plaintext result to dashboard

Usage:
    pip install fastapi uvicorn requests tenseal cryptography
    python bridge_api.py

Then open dashboard.html in your browser.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tenseal as ts
import requests
import os
from base64 import b64decode

app = FastAPI(title="Hospital Bridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CLOUD_URL     = "https://hospital-ckks-cloud.onrender.com"
SECRET_DIR    = r"D:\CNS PROJECT\encrypt\secret_key_LOCAL_ONLY"
CONTEXT_DIR   = r"D:\CNS PROJECT\encrypt\context_output"
ENCRYPTED_DIR = r"D:\CNS PROJECT\encrypt\encrypted_output"

# Cache: track what's already uploaded to cloud this session
uploaded_context  = False
uploaded_datasets = set()


# ─────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    dataset:     str
    column:      str
    operation:   str
    growth_rate: Optional[float] = 0.083


# ─────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────
@app.get("/health")
def health():
    try:
        r = requests.get(f"{CLOUD_URL}/health", timeout=30)
        cloud_data = r.json()
        return {
            "bridge_status"    : "online",
            "cloud_status"     : cloud_data.get("status"),
            "datasets_on_cloud": cloud_data.get("datasets_loaded", []),
            "context_on_cloud" : cloud_data.get("context_loaded"),
        }
    except Exception as e:
        return {"bridge_status": "online", "cloud_status": "unreachable", "error": str(e)}


# ─────────────────────────────────────────
#  UPLOAD CONTEXT (if not already uploaded)
# ─────────────────────────────────────────
def ensure_context_uploaded():
    global uploaded_context
    if uploaded_context:
        return

    path = os.path.join(CONTEXT_DIR, "public_context.tenseal")
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Public context file not found. Run encrypt_data.py first.")

    print("⏳ Uploading public context to cloud (this may take 60s)...")
    with open(path, "rb") as f:
        r = requests.post(
            f"{CLOUD_URL}/upload/context",
            files={"file": ("public_context.tenseal", f, "application/octet-stream")},
            timeout=180,       # ← increased from 30 to 180
        )
    r.raise_for_status()
    uploaded_context = True
    print("✅ Public context uploaded to cloud")


# ─────────────────────────────────────────
#  UPLOAD DATASET (if not already uploaded)
# ─────────────────────────────────────────
def ensure_dataset_uploaded(dataset_name):
    global uploaded_datasets
    if dataset_name in uploaded_datasets:
        return

    fname = f"{dataset_name}_encrypted.pkl"
    path  = os.path.join(ENCRYPTED_DIR, fname)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Encrypted file not found: {fname}. Run encrypt_data.py first.")

    print(f"⏳ Uploading {fname} to cloud...")
    with open(path, "rb") as f:
        r = requests.post(
            f"{CLOUD_URL}/upload/dataset",
            files={"file": (fname, f, "application/octet-stream")},
            timeout=300,       # ← increased from 60 to 300
        )
    r.raise_for_status()
    uploaded_datasets.add(dataset_name)
    print(f"✅ Dataset '{dataset_name}' uploaded to cloud")


# ─────────────────────────────────────────
#  DECRYPT RESULT
# ─────────────────────────────────────────
def decrypt_result(encrypted_b64: str) -> float:
    secret_path = os.path.join(SECRET_DIR, "secret_context.tenseal")
    if not os.path.exists(secret_path):
        raise HTTPException(status_code=500, detail="Secret key not found. Run encrypt_data.py first.")

    with open(secret_path, "rb") as f:
        secret_context = ts.context_from(f.read())

    enc_bytes  = b64decode(encrypted_b64)
    enc_result = ts.ckks_vector_from(secret_context, enc_bytes)
    decrypted  = enc_result.decrypt()
    return float(decrypted[0])


# ─────────────────────────────────────────
#  QUERY ENDPOINT
# ─────────────────────────────────────────
@app.post("/query")
def query(req: QueryRequest):
    try:
        ensure_context_uploaded()
        ensure_dataset_uploaded(req.dataset)

        r = requests.post(
            f"{CLOUD_URL}/query",
            json={
                "dataset"    : req.dataset,
                "column"     : req.column,
                "operation"  : req.operation,
                "growth_rate": req.growth_rate,
            },
            timeout=180,       # ← increased from 60 to 180
        )
        r.raise_for_status()
        response = r.json()

        decrypted_value = decrypt_result(response["encrypted_result"])

        return {
            "dataset"     : req.dataset,
            "column"      : req.column,
            "operation"   : req.operation,
            "value"       : decrypted_value,
            "row_count"   : response["row_count"],
            "privacy_note": "Result computed on encrypted data. Decrypted locally.",
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Cloud server timed out. Render is waking up — wait 30 seconds and click Retry.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
#  RUN ALL QUERIES AT ONCE
# ─────────────────────────────────────────
@app.get("/analytics")
def run_all_analytics():
    queries = [
        {"dataset": "encounters",  "column": "TOTAL_CLAIM_COST",    "operation": "sum"},
        {"dataset": "encounters",  "column": "TOTAL_CLAIM_COST",    "operation": "average"},
        {"dataset": "encounters",  "column": "TOTAL_CLAIM_COST",    "operation": "variance"},
        {"dataset": "encounters",  "column": "PAYER_COVERAGE",      "operation": "sum"},
        {"dataset": "medications", "column": "TOTALCOST",           "operation": "sum"},
        {"dataset": "medications", "column": "TOTALCOST",           "operation": "average"},
        {"dataset": "procedures",  "column": "BASE_COST",           "operation": "sum"},
        {"dataset": "procedures",  "column": "BASE_COST",           "operation": "projected_growth"},
        {"dataset": "patients",    "column": "HEALTHCARE_EXPENSES", "operation": "average"},
        {"dataset": "patients",    "column": "HEALTHCARE_EXPENSES", "operation": "risk_score"},
        {"dataset": "payers",      "column": "REVENUE",             "operation": "sum"},
        {"dataset": "payers",      "column": "AMOUNT_COVERED",      "operation": "sum"},
    ]

    results = []
    errors  = []

    # Upload context once
    try:
        ensure_context_uploaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context upload failed: {e}")

    for q in queries:
        try:
            ensure_dataset_uploaded(q["dataset"])
            r = requests.post(
                f"{CLOUD_URL}/query",
                json=q,
                timeout=180,   # ← increased from 60 to 180
            )
            r.raise_for_status()
            response        = r.json()
            decrypted_value = decrypt_result(response["encrypted_result"])
            results.append({
                "dataset"  : q["dataset"],
                "column"   : q["column"],
                "operation": q["operation"],
                "value"    : decrypted_value,
            })
            print(f"✅ {q['operation'].upper()} [{q['dataset']}.{q['column']}] = {decrypted_value:,.2f}")
        except Exception as e:
            errors.append({"query": q, "error": str(e)})
            print(f"❌ Failed {q}: {e}")

    return {
        "results"     : results,
        "errors"      : errors,
        "privacy_note": "All computations on encrypted data. Decrypted on hospital machine only.",
    }


# ─────────────────────────────────────────
#  LIST AVAILABLE DATASETS
# ─────────────────────────────────────────
@app.get("/datasets")
def list_datasets():
    files = [f.replace("_encrypted.pkl", "") for f in os.listdir(ENCRYPTED_DIR) if f.endswith("_encrypted.pkl")]
    manifest_path = os.path.join(ENCRYPTED_DIR, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
    return {"datasets": files, "manifest": manifest}


# ─────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  HOSPITAL BRIDGE API")
    print("  Running on http://localhost:5000")
    print("  Dashboard can now call this API")
    print("=" * 50)
    uvicorn.run("bridge_api:app", host="0.0.0.0", port=5000, reload=False)