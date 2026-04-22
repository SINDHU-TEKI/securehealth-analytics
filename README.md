# SecureHealth Analytics

SecureHealth Analytics is a privacy-preserving healthcare data analytics system that uses homomorphic encryption to perform computations on encrypted data.

Features
- Secure encryption using CKKS (TenSEAL)
- AES-256 encryption for text data
- Cloud-based computation on encrypted data
- Dashboard for result visualization

## Technologies Used
- Python
- TenSEAL
- FastAPI
- Cryptography (AES)
- HTML/CSS

## How to Run

1. Install dependencies:
pip install tenseal fastapi uvicorn cryptography requests

2. Run encryption:
python encrypt_data.py

3. Start cloud server:
uvicorn cloud_server:app --host 0.0.0.0 --port 8000

4. Run bridge API:
python bridge_api.py

5. Open dashboard.html in browser

## ⚠️ Note
Sensitive files such as secret keys and encrypted datasets are not included for security reasons.
