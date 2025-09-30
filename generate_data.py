import pandas as pd
from faker import Faker
import random
from datetime import timedelta
import os

print("--- Generating Expanded & Enriched Transaction Data for Project Aegis ---")

# Initialize Faker
fake = Faker()

# --- Configuration ---
NUM_ACCOUNTS = 1000
NUM_TRANSACTIONS = 20000
OUTPUT_DIR = 'data/raw'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'simulated_transactions.csv')

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Generate Accounts ---
print(f"Generating {NUM_ACCOUNTS} accounts...")
accounts_data = []
for i in range(NUM_ACCOUNTS):
    accounts_data.append({
        'account_id': 1000 + i,
        'customer_name': fake.name(),
        'account_created_date': fake.date_time_between(start_date='-5y', end_date='-1y'),
    })
accounts_df = pd.DataFrame(accounts_data)
account_ids = list(accounts_df['account_id'])

# --- Generate Transactions ---
transactions_data = []
print(f"Generating {NUM_TRANSACTIONS} baseline transactions...")
transaction_types = ['transfer', 'payment', 'deposit']
merchants = ['Amazon', 'Starbucks', 'Walmart', 'TechSolutions Inc.', 'Global Exports LLC']

for i in range(NUM_TRANSACTIONS):
    sender, receiver = random.sample(account_ids, 2)
    transaction_time = fake.date_time_between(start_date='-1y', end_date='now')
    ttype = random.choice(transaction_types)
    country = fake.country_code()

    transactions_data.append({
        'transaction_id': 100000 + i,
        'sender_account_id': sender,
        'receiver_account_id': receiver,
        'timestamp': transaction_time,
        'amount': round(random.uniform(10.0, 5000.0), 2),
        'transaction_type': ttype,
        'location_country': country,
        'merchant_details': random.choice(merchants) if ttype == 'payment' else None,
        'is_foreign_transaction': True if country != 'US' else False
    })

# --- Engineer a Smurfing (Structuring) Pattern ---
print("Engineering a 'Smurfing' pattern...")
smurf_sender = random.choice(account_ids)
num_smurf_transactions = 50
smurf_receivers = random.sample([aid for aid in account_ids if aid != smurf_sender], num_smurf_transactions)
base_time = fake.date_time_between(start_date='-3M', end_date='-2M')

for i, receiver in enumerate(smurf_receivers):
    transactions_data.append({
        'transaction_id': 200000 + i,
        'sender_account_id': smurf_sender,
        'receiver_account_id': receiver,
        'timestamp': base_time + timedelta(hours=i*2 + random.randint(0,60)),
        'amount': round(random.uniform(100.0, 800.0), 2),
        'transaction_type': 'deposit',
        'location_country': 'US',
        'merchant_details': None,
        'is_foreign_transaction': False
    })

# --- Engineer a Layering (Circular) Pattern ---
print("Engineering a 'Layering' pattern...")
layering_accounts = random.sample(account_ids, 4) # A -> B -> C -> D -> A
base_time = fake.date_time_between(start_date='-6M', end_date='-5M')
layering_amount = round(random.uniform(20000.0, 50000.0), 2)
for i in range(len(layering_accounts)):
    sender = layering_accounts[i]
    receiver = layering_accounts[(i + 1) % len(layering_accounts)]
    transactions_data.append({
        'transaction_id': 300000 + i,
        'sender_account_id': sender,
        'receiver_account_id': receiver,
        'timestamp': base_time + timedelta(minutes=i*15 + random.randint(1,5)),
        'amount': layering_amount - round(random.uniform(50, 200), 2),
        'transaction_type': 'transfer',
        'location_country': 'CY', # Cyprus - common for layering
        'merchant_details': None,
        'is_foreign_transaction': True
    })

# --- Engineer a Pass-Through / Rapid Movement Pattern ---
print("Engineering a 'Pass-Through' pattern...")
pass_through_accounts = random.sample(account_ids, 3) # External -> A -> B -> External
base_time = fake.date_time_between(start_date='-2M', end_date='-1M')
pass_through_amount = round(random.uniform(100000.0, 250000.0), 2)
# Step 1: Money comes in from a high-risk country
transactions_data.append({
    'transaction_id': 400000,
    'sender_account_id': 9999, # External/untracked source
    'receiver_account_id': pass_through_accounts[0],
    'timestamp': base_time,
    'amount': pass_through_amount,
    'transaction_type': 'deposit',
    'location_country': 'PA', # Panama
    'merchant_details': None,
    'is_foreign_transaction': True
})
# Step 2: Money is moved immediately
transactions_data.append({
    'transaction_id': 400001,
    'sender_account_id': pass_through_accounts[0],
    'receiver_account_id': pass_through_accounts[1],
    'timestamp': base_time + timedelta(minutes=random.randint(10, 30)), # Very fast
    'amount': pass_through_amount * 0.99, # Small fee/change
    'transaction_type': 'transfer',
    'location_country': 'US',
    'merchant_details': None,
    'is_foreign_transaction': False
})


# --- Create and Save DataFrame ---
transactions_df = pd.DataFrame(transactions_data)
transactions_df.sort_values(by='timestamp', inplace=True)

print(f"\nSaving data to '{OUTPUT_FILE}'...")
transactions_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Success! {len(transactions_df)} total transactions were generated.")
print("The file is now located in your 'data/raw/' folder.")