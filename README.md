# Credit Ledger

FastAPI + XGBoost loan-default predictor with a small web frontend.

## Deploy to Render (about 5 minutes)

1. Create a new GitHub repository and push **the contents of this folder** (so `main.py`
   and `render.yaml` sit at the repo root, not inside a subfolder).

       git init
       git add .
       git commit -m "Credit Ledger"
       git branch -M main
       git remote add origin https://github.com/<you>/credit-ledger.git
       git push -u origin main

2. In the Render dashboard: **New +  ->  Blueprint**, connect the repo, and click **Apply**.
   Render reads `render.yaml` and builds the service (first build takes a few minutes).

3. When the deploy shows **Live**, your link is `https://credit-ledger.onrender.com`
   (or `https://credit-ledger-xxxx.onrender.com` if that name was taken - the exact URL
   is shown at the top of the service page).

Free plan note: the service sleeps after ~15 minutes idle, so the first request afterwards
takes roughly 30-60 seconds to wake it up.

## Run locally

    pip install -r requirements.txt
    uvicorn main:app --reload

Open http://127.0.0.1:8000

## Endpoints

- `GET  /`        web form
- `GET  /health`  health check (used by Render)
- `POST /predict` JSON body with the 11 applicant fields -> probability, threshold, risk label
- `GET  /docs`    interactive API docs

## Version pins matter

`credit_risk_model.pkl` was saved with scikit-learn 1.6.1 and XGBoost; keep those pins in
`requirements.txt` or loading the model can break. Python is pinned to 3.11.9 via
`.python-version` and `PYTHON_VERSION` in `render.yaml`.
