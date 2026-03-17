import pandas as pd
import json
import time
from kafka import KafkaProducer

#------------------------------------------------------------------- Load Paysim CSV
print("Loading Paysim dataset. . .")
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
print(f"Loaded {len(df)} transactions")

###----------------------------------------------------------------- Feature engineering 
df['type_encoded'] = df['type'].map({
    'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4
})

# check discrepancies between before and after customer who started the transaction, no error = 0
# large positive value means unjustified extra amount after the transaction is done
# large negative value means unjustified amount lost during the transaction
df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']

# check discrepancies between before and after of recipient balance, no error = 0
# large positive value means unjustified money lost during the transaction
# large negative value means unjustified extra money received.
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

# flag CASH_OUT and TRANSFER
df['is_risky_type'] = df['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)

# flag 0 balance
df['orig_zero_balance'] = (df['newbalanceOrig'] == 0).astype(int)
df['dest_zero_balance'] = (df['newbalanceDest'] == 0).astype(int)

# amount to balance ratio, how large is this transaction relative to account
df['amount_to_orig_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)


#------------------------------------------------------------------ Kafka producer
producer = KafkaProducer(
    bootstrap_servers = 'localhost:9092',
    value_serializer = lambda v:json.dumps(v).encode('utf-8')
)

print("Starting to stream transactions. . .")
features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'type_encoded',
    'errorBalanceOrg', 'errorBalanceDest', 'is_risky_type',
    'orig_zero_balance', 'dest_zero_balance', 'amount_to_orig_ratio'
]

for idx, row in df.iterrows():
    message = row[features].to_dict()
    message['isFraud'] = int(row['isFraud']) # label for monitoring
    message['transaction_id'] = idx

    producer.send('bank-transactions', message)

    # Simulate 1 transaction per 10ms (100transactions/second)
    time.sleep(0.05)

    if idx % 1000 == 0:
        print(f"Streamed {idx} transactions. . .")

producer.flush()
print("Done Streaming.")