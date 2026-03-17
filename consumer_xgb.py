import json
import joblib
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

# Load pre-trained XGBoost model
print("Loading XGBoost model. . .")
model = joblib.load('xgb_fraud_model.pkl')
print("Model loaded successfully")


# Features
features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'type_encoded',
    'errorBalanceOrg', 'errorBalanceDest', 'is_risky_type',
    'orig_zero_balance', 'dest_zero_balance', 'amount_to_orig_ratio'
]

# Consumer - reads from bank-transactios topic
consumer = KafkaConsumer(
    'bank-transactions',
    bootstrap_servers = 'localhost:9092',
    value_deserializer = lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='xgb-consumer-group'
)

# Producer - sends fraud alerts to xgb-alerts topic
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer= lambda v: json.dumps(v).encode('utf-8')
)

print("Listening for transaction. . .")

# Counters for monitoring
total = 0
fraud_detected = 0
correct = 0

# optimal threshold
FRAUD_THRESHOLD = 0.86

# scoring loop
for message in consumer:
    transaction = message.value

    # Extract feature in correct order
    X = np.array([[transaction[f] for f in features]])

    # Score with XGBoost
    fraud_proba = model.predict_proba(X)[0][1]
    prediction = int(fraud_proba >= FRAUD_THRESHOLD)

    # Track accuracy
    actual = transaction['isFraud']
    total += 1
    if prediction == 1:
        fraud_detected +=1
    if prediction == actual:
        correct += 1

    # Send to xgb-alerts topic
    alert = {
        'transaction_id': int(transaction['transaction_id']),
        'fraud_probability': float(fraud_proba),
        'prediction': int(prediction),
        'actual': int(actual),
        'amount': float(transaction['amount']),
        'type_encoded': int(transaction['type_encoded'])
    }
    producer.send('xgb-alerts',alert)

    # Print every 100 transactions
    if total % 100 == 0:
        accuracy = correct / total * 100
        print(f"[XGB] Processed: {total} | "
              f"Fraud detected: {fraud_detected} | "
              f"Accuracy: {accuracy:.2f}%")