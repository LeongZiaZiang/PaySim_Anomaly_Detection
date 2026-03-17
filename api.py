from fastapi import FastAPI
from kafka import KafkaConsumer
import json
import threading

app = FastAPI(title="Fraud Detection API")

# Storage for latest alerts
xgb_alerts = []
river_alerts = []
total_fraud_xgb = 0
total_anomaly_river = 0
MAX_ALERTS = 1000 # keep last 1000 alerts in memory

# --- Background Kafka consumers ---

def consume_xgb():
    consumer = KafkaConsumer(
        'xgb-alerts',
        bootstrap_servers = 'localhost:9092',
        value_deserializer = lambda m:json.loads(m.decode('utf-8')),
        auto_offset_reset = 'earliest',
        group_id = 'api-xgb-group'
    )

    global total_fraud_xgb
    
    for message in consumer:
        xgb_alerts.append(message.value)
        if message.value['prediction'] == 1 and message.value['actual'] == 1:
            total_fraud_xgb += 1
        if len(xgb_alerts) > MAX_ALERTS:
            xgb_alerts.pop(0)

def consume_river():
    consumer = KafkaConsumer(
        'river-alerts',
        bootstrap_servers = 'localhost:9092',
        value_deserializer = lambda m:json.loads(m.decode('utf-8')),
        auto_offset_reset = 'earliest',
        group_id = 'api-river-group'
    )

    global total_anomaly_river
    
    for message in consumer:
        river_alerts.append(message.value)
        if message.value['prediction'] == 1 and message.value['actual'] == 1:
            total_anomaly_river += 1
        if len(river_alerts) > MAX_ALERTS:
            river_alerts.pop(0)

# Start background threads when API starts
@app.on_event("startup")
def startup_event():
    threading.Thread(target=consume_xgb, daemon=True).start()
    threading.Thread(target=consume_river, daemon=True).start()
    print("Background Kafka consumers started")

# --- API Endpoints ---

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/xgb/alerts")
def get_xgb_alerts(limit: int = 50):
    """Get Latest XGBoost fraud alerts"""
    fraud_only = [a for a in xgb_alerts if a['prediction'] == 1]
    return {
        "model": "XGBoost",
        "total_processed": len(xgb_alerts),
        "total_fraud_detected": len(fraud_only),
        "latest_alerts": fraud_only[-limit:]
    }

@app.get("/river/alerts")
def get_river_alerts(limit: int = 50):
    """Get latest River anomaly alerts"""
    anomalies_only = [a for a in river_alerts if a['prediction'] == 1]
    return{
        'model': "River HalfSpaceTrees",
        "total_processed": len(river_alerts),
        "total_anomalies_detected": len(anomalies_only),
        "latest_alerts": anomalies_only[-limit:]
    }

@app.get("/summary")
def get_summary():
    """Compare both models side by side"""
    xgb_fraud = [a for a in xgb_alerts if a['prediction'] == 1]
    river_fraud = [a for a in river_alerts if a['prediction'] == 1]

    return{
        "xgboost": {
            "total_processed": len(xgb_alerts),
            "fraud_detected": len(xgb_fraud),
            "fraud_rate": round(len(xgb_fraud) / max(len(xgb_alerts), 1) * 100, 4),
            "total_fraud": total_fraud_xgb
        },
        "river": {
            "total_processed": len(river_alerts),
            "anomalies_detected": len(river_fraud),
            "anomaly_rate": round(len(river_fraud) / max(len(river_alerts), 1) * 100, 4),
            "total_anomaly": total_anomaly_river
        }
    }

@app.get("/xgb/stats")
def get_xgb_stats():
    """XGBoost model performance stats"""
    if not xgb_alerts:
        return {"message":"No data yet"}

    correct = sum(1 for a in xgb_alerts if a['prediction'] == a['actual'])
    fraud_caught = sum(1 for a in xgb_alerts if a['prediction'] == 1 and a['actual'] == 1)
    fraud_missed = sum(1 for a in xgb_alerts if a['prediction'] == 0 and a['actual'] == 1)
    false_alarms = sum(1 for a in xgb_alerts if a['prediction'] == 1 and a['actual'] == 0)

    return {
        "total_processed": len(xgb_alerts),
        "accuracy": round(correct / len(xgb_alerts) * 100, 4),
        "fraud_caught": fraud_caught,
        "fraud_missed": fraud_missed,
        "false_alarms": false_alarms
    }

@app.get("/river/stats")
def get_river_stats():
    """River model performance stats"""
    if not river_alerts:
        return {"message": "No data yet"}
    
    correct = sum(1 for a in river_alerts if a['prediction'] == a['actual'])
    fraud_caught = sum(1 for a in river_alerts if a['prediction'] == 1 and a['actual'] == 1)
    fraud_missed = sum(1 for a in river_alerts if a['prediction'] == 0 and a['actual'] == 1)
    false_alarms = sum(1 for a in river_alerts if a['prediction'] == 1 and a['actual'] == 0)

    return {
        "total_processed": len(river_alerts),
        "accuracy": round(correct / len(river_alerts) * 100, 4),
        "fraud_caught": fraud_caught,
        "fraud_missed": fraud_missed,
        "false_alarms": false_alarms
    }