import pandas as pd
import numpy as np
from faker import Faker
from sklearn.ensemble import IsolationForest
import random
from datetime import datetime, timedelta

# Initialize Faker for data generation
fake = Faker()

print("--- Aegis-DA Prototype Script ---")

# --- Step 1: Data Simulation ---
print("[1/4] Simulating accounts and transactions...")

# Configuration
NUM_ACCOUNTS = 500
NUM_NORMAL_TRANSACTIONS = 10000

# Generate Accounts
accounts_data = []
for i in range(NUM_ACCOUNTS):
    accounts_data.append({
        'account_id': 1000 + i,
        'customer_name': fake.name(),
        'account_created_date': fake.date_time_between(start_date='-5y', end_date='-1y'),
    })
accounts_df = pd.DataFrame(accounts_data)

# Generate Normal Transactions
transactions_data = []
for _ in range(NUM_NORMAL_TRANSACTIONS):
    sender, receiver = random.sample(list(accounts_df['account_id']), 2)
    transactions_data.append({
        'sender_account_id': sender,
        'receiver_account_id': receiver,
        'timestamp': fake.date_time_between(start_date='-1y', end_date='now'),
        'amount': round(random.uniform(10.0, 5000.0), 2)
    })

# --- Engineer Suspicious Patterns ---

# Pattern 1: Layering (Circular Transactions)
# A -> B -> C -> A
layering_accounts = random.sample(list(accounts_df['account_id']), 3)
base_time = fake.date_time_between(start_date='-1M', end_date='now')
for i in range(len(layering_accounts)):
    sender = layering_accounts[i]
    receiver = layering_accounts[(i + 1) % len(layering_accounts)]
    transactions_data.append({
        'sender_account_id': sender,
        'receiver_account_id': receiver,
        'timestamp': base_time + timedelta(minutes=i*5), # Small time gaps
        'amount': round(random.uniform(9000.0, 10000.0), 2) # High, round amounts
    })

# Pattern 2: Smurfing (One account sends small amounts to many)
smurf_sender = random.choice(accounts_df['account_id'])
num_smurf_transactions = 20
receivers = random.sample(list(accounts_df[accounts_df['account_id'] != smurf_sender]['account_id']), num_smurf_transactions)
base_time = fake.date_time_between(start_date='-1M', end_date='now')
for i, receiver in enumerate(receivers):
    transactions_data.append({
        'sender_account_id': smurf_sender,
        'receiver_account_id': receiver,
        'timestamp': base_time + timedelta(hours=i),
        'amount': round(random.uniform(100.0, 500.0), 2) # Small, structured amounts
    })
    
transactions_df = pd.DataFrame(transactions_data)
transactions_df['transaction_id'] = range(1, len(transactions_df) + 1)


# --- Step 2: Feature Engineering ---
print("[2/4] Performing feature engineering...")

# Merge data to get account creation dates
df = pd.merge(transactions_df, accounts_df, left_on='sender_account_id', right_on='account_id', suffixes=('', '_sender'))
df.rename(columns={'account_created_date': 'sender_creation_date'}, inplace=True)
df.drop(columns=['customer_name', 'account_id'], inplace=True)

# Ensure timestamps are datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['sender_creation_date'] = pd.to_datetime(df['sender_creation_date'])

# Feature 1: Account Age at Transaction
df['account_age_days'] = (df['timestamp'] - df['sender_creation_date']).dt.days

# Feature 2: Transaction Hour
df['transaction_hour'] = df['timestamp'].dt.hour
df['is_odd_hour'] = df['transaction_hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# Feature 3: Is Round Number Transaction
df['is_round_amount'] = df['amount'].apply(lambda x: 1 if x % 1000 == 0 and x > 0 else 0)

# Clean up for modeling
df.fillna(0, inplace=True) # Handle potential NaNs

# --- Step 3: Anomaly Detection Model ---
print("[3/4] Applying Isolation Forest for anomaly detection...")

# Select features for the model
features = ['amount', 'account_age_days', 'is_odd_hour', 'is_round_amount']
X = df[features]

# Initialize and train the model
# Contamination is the expected proportion of anomalies in the data
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Get anomaly scores (-1 for anomalies, 1 for inliers)
df['anomaly_score_raw'] = model.decision_function(X)
# Normalize score to be more intuitive (lower is more anomalous)
df['anomaly_score'] = (df['anomaly_score_raw'] - df['anomaly_score_raw'].min()) / (df['anomaly_score_raw'].max() - df['anomaly_score_raw'].min())


# --- Step 4: Final Output ---
print("[4/4] Generating final CSV for Power BI...")

# Select and rename columns for the final report
final_df = df[[
    'transaction_id',
    'sender_account_id',
    'receiver_account_id',
    'timestamp',
    'amount',
    'account_age_days',
    'is_odd_hour',
    'is_round_amount',
    'anomaly_score'
]].sort_values(by='anomaly_score', ascending=True)

# Save to CSV
output_filename = 'enriched_transactions_for_powerbi.csv'
final_df.to_csv(output_filename, index=False)

print(f"\n✅ Success! Prototype finished.")
print(f"Output file created: {output_filename}")
print("\n--- Top 5 Most Anomalous Transactions ---")
print(final_df.head())
print("\nThis file is now ready to be imported into Power BI.")