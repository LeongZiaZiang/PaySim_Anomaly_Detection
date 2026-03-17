import json
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from river  import anomaly, preprocessing

# River online learning model
print("Initialising River HalfSpaceTrees model. . .")
scaler = preprocessing.StandardScaler()
model = anomaly.HalfSpaceTrees(
    n_trees=10,
    height=8,
    window_size=250,
    seed=42
)

# Features
features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'type_encoded',
    'errorBalanceOrg', 'errorBalanceDest', 'is_risky_type',
    'orig_zero_balance', 'dest_zero_balance', 'amount_to_orig_ratio'
]

# Consumer - reads from bank-transactions topic
consumer = KafkaConsumer(
    'bank-transactions',
    bootstrap_servers = 'localhost:9092',
    value_deserializer = lambda m:json.loads(m.decode('utf-8')),
    auto_offset_reset = 'earliest',
    group_id = 'river-consumer-group'
)

# Producer - sends results to river-alerts topic
producer = KafkaProducer(
    bootstrap_servers = 'localhost:9092',
    value_serializer = lambda v: json.dumps(v).encode('utf-8')
)

print("Listening for transactions. . .")

# Counters
total = 0
fraud_detected = 0
correct = 0

# Anomaly threshold - tune this based on score distribution
ANOMALY_THRESHOLD = 0.92

for message in consumer:
    transaction = message.value

    # Build feature dictionary for River (expects dict not array)
    x = {f: transaction[f] for f in features}

    # Scale features
    scaler.learn_one(x)
    x_scaled = scaler.transform_one(x)

    # Score BEFORE learning (score first, then update)
    anomaly_score = model.score_one(x_scaled)
    prediction = int(anomaly_score >= ANOMALY_THRESHOLD)

    # Update model with this transaction
    model.learn_one(x_scaled)

    # Track performance
    actual = transaction['isFraud']
    total += 1
    if prediction == 1:
        fraud_detected += 1
    if prediction == actual:
        correct += 1

    # send to river-alerts topic
    alert = {
        'transaction_id': int(transaction['transaction_id']),
        'anomaly_score': float(anomaly_score),
        'prediction': int(prediction),
        'actual': int(actual),
        'amount': float(transaction['amount']),
        'type_encoded': int(transaction['type_encoded'])
    }
    producer.send('river-alerts',alert)

    # Print every 100 transactions
    if total % 100 == 0:
        accuracy = correct / total * 100
        print(f"[RIVER] Processed: {total} | "
              f"Anomalies detected: {fraud_detected} | "
              f"Accuracy: {accuracy:.2f}%")