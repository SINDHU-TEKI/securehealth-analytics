"""
STEP 1 (Updated) — Encrypt ALL Columns from CSV Datasets
=========================================================
- Numeric columns  → CKKS homomorphic encryption (TenSEAL)
                     Cloud CAN perform computations on these
- Text/String cols → AES-256 symmetric encryption (cryptography)
                     Cloud CANNOT read these, only stores them

Install dependencies:
    pip install tenseal pandas cryptography

Usage:
    python encrypt_data.py
"""

import os
import json
import pickle
import base64
import tenseal as ts
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────

DATA_DIR = r"D:\CNS PROJECT\data"  # ← folder with your CSV files
ENCRYPTED_DIR = "./encrypted_output"
CONTEXT_DIR   = "./context_output"
SECRET_DIR    = "./secret_key_LOCAL_ONLY"

os.makedirs(ENCRYPTED_DIR, exist_ok=True)
os.makedirs(CONTEXT_DIR,   exist_ok=True)
os.makedirs(SECRET_DIR,    exist_ok=True)

# CKKS parameters
POLY_MOD_DEGREE     = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE        = 2 ** 40

# Columns that look numeric but are actually IDs — encrypt as TEXT
ID_COLUMNS = {
    "CODE", "REASONCODE", "BODYSITE_CODE",
    "ZIP", "PHONE", "START_YEAR", "END_YEAR",
    "SSN", "DRIVERS", "PASSPORT"
}


# ─────────────────────────────────────────
#  GENERATE CKKS CONTEXT
# ─────────────────────────────────────────

def generate_ckks_context():
    print("🔑 Generating CKKS context...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.global_scale = GLOBAL_SCALE

    secret_bytes = context.serialize(save_secret_key=True)
    context.make_context_public()
    public_bytes = context.serialize()

    print("✅ CKKS context generated (128-bit security)")
    return secret_bytes, public_bytes


# ─────────────────────────────────────────
#  GENERATE AES-256 KEY
# ─────────────────────────────────────────

def generate_aes_key():
    print("🔑 Generating AES-256 key for text columns...")
    aes_key = secrets.token_bytes(32)
    print("✅ AES-256 key generated")
    return aes_key


# ─────────────────────────────────────────
#  AES-256 ENCRYPT/DECRYPT
# ─────────────────────────────────────────

def aes_encrypt(value: str, aes_key: bytes) -> str:
    if not value or str(value).strip() == "" or str(value) == "nan":
        return ""
    aesgcm = AESGCM(aes_key)
    nonce  = secrets.token_bytes(12)
    ct     = aesgcm.encrypt(nonce, str(value).encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("utf-8")


def aes_decrypt(encrypted_value: str, aes_key: bytes) -> str:
    if not encrypted_value:
        return ""
    raw    = base64.b64decode(encrypted_value.encode("utf-8"))
    nonce  = raw[:12]
    ct     = raw[12:]
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ct, None).decode("utf-8")


# ─────────────────────────────────────────
#  CKKS ENCRYPT NUMERIC COLUMN
# ─────────────────────────────────────────

def ckks_encrypt_column(values, context):
    float_values = [float(v) for v in values]
    enc_vector   = ts.ckks_vector(context, float_values)
    return enc_vector.serialize()


# ─────────────────────────────────────────
#  DETECT COLUMN TYPE
# ─────────────────────────────────────────

def detect_column_type(col_name, series):
    if col_name.upper() in ID_COLUMNS:
        return "text"
    numeric       = pd.to_numeric(series.dropna(), errors="coerce")
    non_null      = series.dropna().shape[0]
    numeric_count = numeric.dropna().shape[0]
    if non_null > 0 and (numeric_count / non_null) > 0.8:
        return "numeric"
    return "text"


# ─────────────────────────────────────────
#  ENCRYPT ALL DATASETS
# ─────────────────────────────────────────

def encrypt_all_datasets(ckks_context, aes_key):
    manifest  = {}
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"\n📂 Found {len(csv_files)} CSV files in {DATA_DIR}")

    for csv_file in csv_files:
        dataset_name = csv_file.replace(".csv", "")
        path         = os.path.join(DATA_DIR, csv_file)

        print(f"\n{'='*50}")
        print(f"📊 Processing: {dataset_name}")
        print(f"{'='*50}")

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  ❌ Could not read {csv_file}: {e}")
            continue

        print(f"  📂 Loaded → {len(df)} rows, {len(df.columns)} columns")

        encrypted_data = {
            "dataset"         : dataset_name,
            "row_count"       : len(df),
            "numeric_columns" : {},
            "text_columns"    : {},
            "col_types"       : {},
            "col_meta"        : {},
        }

        numeric_count = 0
        text_count    = 0

        for col in df.columns:
            col_type = detect_column_type(col, df[col])
            encrypted_data["col_types"][col] = col_type

            if col_type == "numeric":
                values = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                if not values:
                    continue
                print(f"  🔐 [CKKS]    {col} ({len(values)} values)...")
                enc_bytes = ckks_encrypt_column(values, ckks_context)
                encrypted_data["numeric_columns"][col] = enc_bytes
                encrypted_data["col_meta"][col] = {
                    "count": len(values),
                    "min"  : float(min(values)),
                    "max"  : float(max(values)),
                }
                numeric_count += 1

            else:
                print(f"  🔒 [AES-256] {col} ({len(df[col])} values)...")
                encrypted_values = [aes_encrypt(str(v), aes_key) for v in df[col]]
                encrypted_data["text_columns"][col] = encrypted_values
                text_count += 1

        out_path = os.path.join(ENCRYPTED_DIR, f"{dataset_name}_encrypted.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(encrypted_data, f)

        print(f"\n  ✅ Saved → {out_path}")
        print(f"     CKKS columns    : {numeric_count}")
        print(f"     AES-256 columns : {text_count}")

        manifest[dataset_name] = {
            "numeric_columns": list(encrypted_data["numeric_columns"].keys()),
            "text_columns"   : list(encrypted_data["text_columns"].keys()),
            "row_count"      : len(df),
            "file"           : f"{dataset_name}_encrypted.pkl"
        }

    manifest_path = os.path.join(ENCRYPTED_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n📋 Manifest saved → {manifest_path}")
    return manifest


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 50)
    print("  HOSPITAL ENCRYPTION — FULL COLUMN ENCRYPTION")
    print("  Numeric → CKKS  |  Text → AES-256")
    print("=" * 50)

    # 1. Generate keys
    secret_ckks_bytes, public_ckks_bytes = generate_ckks_context()
    aes_key = generate_aes_key()

    # 2. Save secret keys locally
    with open(os.path.join(SECRET_DIR, "secret_context.tenseal"), "wb") as f:
        f.write(secret_ckks_bytes)
    with open(os.path.join(SECRET_DIR, "aes_key.bin"), "wb") as f:
        f.write(aes_key)

    print(f"\n🔒 Secret keys saved → {SECRET_DIR}/")
    print(f"   ⚠️  NEVER upload these to cloud!")

    # 3. Save public context
    with open(os.path.join(CONTEXT_DIR, "public_context.tenseal"), "wb") as f:
        f.write(public_ckks_bytes)
    print(f"☁️  Public context saved → {CONTEXT_DIR}/")

    # 4. Encrypt all datasets
    ckks_context = ts.context_from(secret_ckks_bytes)
    manifest     = encrypt_all_datasets(ckks_context, aes_key)

    # 5. Summary
    total_numeric = sum(len(v["numeric_columns"]) for v in manifest.values())
    total_text    = sum(len(v["text_columns"])    for v in manifest.values())

    print("\n" + "=" * 50)
    print("  ✅ ENCRYPTION COMPLETE")
    print("=" * 50)
    print(f"  Datasets          : {len(manifest)}")
    print(f"  CKKS columns      : {total_numeric}")
    print(f"  AES-256 columns   : {total_text}")
    print(f"  Total columns     : {total_numeric + total_text}")
    print(f"\n  🔒 Secret keys  → {SECRET_DIR}/  ← NEVER UPLOAD")
    print(f"  ➡️  Next: Run client.py to upload to cloud")


if __name__ == "__main__":
    main()