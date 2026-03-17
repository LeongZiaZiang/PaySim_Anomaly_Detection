# Real-Time Fraud Detection System

A real-time transaction fraud detection pipeline that streams PaySim bank transactions through Kafka and scores them simultaneously with two models — a pre-trained XGBoost classifier and an online-learning River HalfSpaceTrees model. Results are served via a FastAPI backend and visualised on a live Streamlit dashboard.

---

## Architecture

```
PaySim CSV
    │
    ▼
producer.py  ──────────►  Kafka topic: bank-transactions
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        consumer_xgb.py                consumer_river.py
        (XGBoost batch model)          (River online model)
                │                               │
                ▼                               ▼
        Kafka: xgb-alerts              Kafka: river-alerts
                │                               │
                └───────────────┬───────────────┘
                                ▼
                            api.py
                        (FastAPI backend)
                                │
                                ▼
                         dashboard.py
                       (Streamlit UI)
```

---

## Models

### XGBoost (Batch)
Pre-trained offline on the full PaySim dataset and loaded at consumer startup. Scores each incoming transaction against a fixed model.

| Metric | Value |
|---|---|
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9988 |
| Recall (Fraud) | 0.9976 |
| Precision (Fraud) | 0.9903 |
| F1 Score | 0.9939 |

> Threshold tuned to 0.86 via precision-recall curve intersection on validation set.

### River HalfSpaceTrees (Online)
Learns and adapts continuously from each transaction as it arrives. No pre-training required — the model updates itself in real time.

| Metric | Value |
|---|---|
| Anomaly threshold | 0.92 |
| Trees | 10 |
| Height | 8 |
| Window size | 250 |

---

## Features

All features are engineered from the raw PaySim columns:

| Feature | Description |
|---|---|
| `step` | Time unit (1 step = 1 hour) |
| `amount` | Transaction amount |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before/after |
| `oldbalanceDest` / `newbalanceDest` | Recipient balance before/after |
| `type_encoded` | Transaction type (label encoded) |
| `errorBalanceOrg` | Balance discrepancy on sender side |
| `errorBalanceDest` | Balance discrepancy on recipient side |
| `is_risky_type` | 1 if CASH_OUT or TRANSFER |
| `orig_zero_balance` | 1 if sender balance is 0 after transaction |
| `dest_zero_balance` | 1 if recipient balance is 0 after transaction |
| `amount_to_orig_ratio` | Transaction size relative to sender balance |

---

## Setup

### 1. Get the dataset

Download the PaySim dataset from Kaggle:  
[https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

Place the CSV file in the project root directory.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Kafka

```bash
docker-compose up -d
```

### 4. Train and save the XGBoost model

Run `02_analysis.ipynb` end to end. This saves `xgb_fraud_model.pkl` to the project root.

### 5. Run the pipeline

Open four terminals and run each component:

```bash
# Terminal 1 — FastAPI backend
uvicorn api:app --reload

# Terminal 2 — XGBoost consumer
python consumer_xgb.py

# Terminal 3 — River consumer
python consumer_river.py

# Terminal 4 — Stream transactions
python producer.py
```

### 6. Open the dashboard

```bash
streamlit run dashboard.py
```

Navigate to `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/summary` | Side-by-side model comparison |
| GET | `/xgb/alerts` | Latest XGBoost fraud alerts |
| GET | `/river/alerts` | Latest River anomaly alerts |
| GET | `/xgb/stats` | XGBoost performance stats |
| GET | `/river/stats` | River performance stats |

Interactive docs available at `http://localhost:8000/docs`

---

## Project Structure

```
├── 01_eda.ipynb          # Exploratory data analysis
├── 02_analysis.ipynb     # Model training, evaluation, threshold tuning
├── producer.py           # Streams transactions to Kafka
├── consumer_xgb.py       # XGBoost scoring consumer
├── consumer_river.py     # River online learning consumer
├── api.py                # FastAPI backend
├── dashboard.py          # Streamlit dashboard
├── docker-compose.yml    # Kafka + Zookeeper setup
└── requirements.txt
```

---

## Dataset

[PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile money transaction dataset simulated using real transaction logs from a mobile money service in Africa. Contains ~6.3M transactions across 30 days with ground truth fraud labels.
