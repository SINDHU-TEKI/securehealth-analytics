import os
import pickle
import requests
import tenseal as ts
from base64 import b64decode

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CLOUD_URL     = "https://hospital-ckks-cloud.onrender.com"

SECRET_DIR    = r"D:\CNS PROJECT\encrypt\secret_key_LOCAL_ONLY"
CONTEXT_DIR   = r"D:\CNS PROJECT\encrypt\context_output"
ENCRYPTED_DIR = r"D:\CNS PROJECT\encrypt\encrypted_output"


# ─────────────────────────────────────────
#  CHECK CLOUD IS ONLINE
# ─────────────────────────────────────────
def check_health():
    print("🌐 Checking cloud server...")
    r = requests.get(f"{CLOUD_URL}/health")
    r.raise_for_status()
    data = r.json()
    print(f"✅ Cloud server online")
    print(f"   Datasets loaded : {data['datasets_loaded']}")
    print(f"   Context loaded  : {data['context_loaded']}")
    return data


# ─────────────────────────────────────────
#  UPLOAD PUBLIC CONTEXT
# ─────────────────────────────────────────
def upload_context():
    print("\n☁️  Uploading public context to cloud...")
    path = os.path.join(CONTEXT_DIR, "public_context.tenseal")
    with open(path, "rb") as f:
        r = requests.post(
            f"{CLOUD_URL}/upload/context",
            files={"file": ("public_context.tenseal", f, "application/octet-stream")}
        )
    r.raise_for_status()
    print(f"✅ {r.json()['message']}")


# ─────────────────────────────────────────
#  UPLOAD ENCRYPTED DATASETS
# ─────────────────────────────────────────
def upload_datasets(datasets=None):
    all_files = [f for f in os.listdir(ENCRYPTED_DIR) if f.endswith("_encrypted.pkl")]

    if datasets:
        all_files = [f"{d}_encrypted.pkl" for d in datasets if os.path.exists(
            os.path.join(ENCRYPTED_DIR, f"{d}_encrypted.pkl")
        )]

    print(f"\n📤 Uploading {len(all_files)} encrypted datasets to cloud...")

    for fname in all_files:
        path = os.path.join(ENCRYPTED_DIR, fname)
        print(f"  Uploading {fname}...")
        with open(path, "rb") as f:
            r = requests.post(
                f"{CLOUD_URL}/upload/dataset",
                files={"file": (fname, f, "application/octet-stream")}
            )
        r.raise_for_status()
        data = r.json()
        print(f"  ✅ {data['message']} | columns: {data['columns']}")


# ─────────────────────────────────────────
#  SEND QUERY & DECRYPT RESULT
# ─────────────────────────────────────────
def query_and_decrypt(dataset, column, operation, growth_rate=0.083):
    print(f"\n🔍 Query: {operation.upper()} of [{column}] in [{dataset}]")
    print(f"   Sending to cloud... (cloud sees only ciphertext)")

    r = requests.post(f"{CLOUD_URL}/query", json={
        "dataset"    : dataset,
        "column"     : column,
        "operation"  : operation,
        "growth_rate": growth_rate,
    })
    r.raise_for_status()
    response = r.json()

    print(f"   ✅ Cloud computed {operation} on encrypted data")
    print(f"   📦 Received encrypted result ({len(response['encrypted_result'])} chars)")

    # Decrypt using secret key
    secret_path = os.path.join(SECRET_DIR, "secret_context.tenseal")
    with open(secret_path, "rb") as f:
        secret_context = ts.context_from(f.read())

    enc_bytes  = b64decode(response["encrypted_result"])
    enc_result = ts.ckks_vector_from(secret_context, enc_bytes)
    decrypted  = enc_result.decrypt()
    final_value = decrypted[0]

    print(f"   🔓 Decrypted result: {final_value:,.4f}")
    return final_value


# ─────────────────────────────────────────
#  FULL ANALYTICS
# ─────────────────────────────────────────
def run_full_analytics():
    print("=" * 60)
    print("  HOSPITAL ANALYTICS — SECURE CKKS PIPELINE")
    print("=" * 60)

    queries = [
        ("encounters",  "TOTAL_CLAIM_COST",    "sum"),
        ("encounters",  "TOTAL_CLAIM_COST",    "average"),
        ("encounters",  "TOTAL_CLAIM_COST",    "variance"),
        ("medications", "TOTALCOST",           "sum"),
        ("medications", "TOTALCOST",           "average"),
        ("procedures",  "BASE_COST",           "sum"),
        ("procedures",  "BASE_COST",           "projected_growth"),
        ("patients",    "HEALTHCARE_EXPENSES", "average"),
        ("patients",    "HEALTHCARE_EXPENSES", "risk_score"),
        ("payers",      "REVENUE",             "sum"),
    ]

    results = {}
    for dataset, column, operation in queries:
        try:
            val = query_and_decrypt(dataset, column, operation)
            key = f"{dataset}.{column}.{operation}"
            results[key] = val
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\n" + "=" * 60)
    print("  ✅ DECRYPTED ANALYTICS RESULTS")
    print("  (All computed on encrypted cloud data)")
    print("=" * 60)
    for key, val in results.items():
        parts = key.split(".")
        print(f"  {parts[2].upper():20s} | {parts[0]:15s} | {parts[1]:25s} | {val:>20,.2f}")

    print("\n🛡️  Privacy preserved: Cloud never saw raw values.")
    return results


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    check_health()
    upload_context()
    upload_datasets(datasets=["encounters", "medications", "procedures", "patients", "payers"])
    run_full_analytics()


if __name__ == "__main__":
    main()