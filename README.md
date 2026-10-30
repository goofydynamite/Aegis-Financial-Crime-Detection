# Project Aegis: Anti-Money-Laundering Detection on Transaction Networks

Aegis ranks financial transactions by money-laundering risk so an investigation team
can review the riskiest first. It is trained and evaluated on the
[IBM AML dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
(Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering
Models*, NeurIPS 2023): 6.9 million transactions, 706k accounts, 0.05% laundering.

## Results (LI-Small, chronological hold-out)

Test period 8–10 Sept 2022: 1,384,828 transactions, 791 laundering (1 in 1,750).

| Model | PR-AUC | ROC-AUC | Precision@100 | Recall@1000 | Recall@5000 |
|---|---|---|---|---|---|
| Legacy prototype (Isolation Forest, per-transaction features) | 0.0005 | 0.456 | 0.02 | 0.004 | 0.006 |
| Isolation Forest, behavioural + network features | 0.0015 | 0.778 | 0.00 | 0.000 | 0.009 |
| **LightGBM, behavioural + network features** | **0.233** | **0.966** | **0.95** | **0.259** | **0.448** |

- **95 of the top 100 alerts are laundering.** 100 random alerts would contain about 0.06.
- The top 5,000 alerts (0.36% of transactions) contain 45% of all laundering.
- Of transactions linked to a known typology, 85% land in the top 5,000. Recall per
  typology ranges from 95% (cycles) to 63% (stacks), in
  [reports/summary.md](reports/summary.md).
- The legacy approach from the first version of this project performs no better than random
  on real-scale data: single-transaction features cannot see network patterns.

Accuracy is not reported: predicting "not laundering" for everything scores 99.95%.

## How it works

```
raw CSV ──ingest──▶ transactions.parquet ──features──▶ features.parquet ──train──▶ model + reports
```

1. **Ingest** ([src/aegis/data.py](src/aegis/data.py)): typed load, deduplication, currency
   conversion to USD (rates estimated from the data's own cross-currency payments),
   integer account IDs, typology labels joined from `*_Patterns.txt`, and removal of the
   dataset's near-empty trailing days (see *Data caveats*).
2. **Features** ([src/aegis/features.py](src/aegis/features.py)): 26 features, each
   computed only from the current and earlier transactions, so the same code can score a
   live stream. A test enforces this.
   - *Sender behaviour*: payment count, 24h velocity and volume, distinct receivers so far,
     time since previous payment, amount relative to the sender's own average.
   - *Receiver behaviour*: the same, from the incoming side (fan-in).
   - *Flow-through*: time since the sender last received money and the ratio of this
     payment to that inflow. These catch pass-through and stack layering.
   - *Hub signals*: how many distinct accounts paid the sender, and how many the receiver
     has paid (gather and scatter nodes).
3. **Train and evaluate** ([src/aegis/model.py](src/aegis/model.py),
   [src/aegis/evaluate.py](src/aegis/evaluate.py)): 60/20/20 chronological split, LightGBM
   with early stopping on validation PR-AUC, two Isolation Forest baselines, and
   precision/recall@k metrics that match how a fixed-size investigation team works.

## Project structure

```
aegis/
├── config.yaml              # dataset, split, model and evaluation settings
├── src/aegis/
│   ├── __main__.py          # CLI: python -m aegis {ingest,features,train,all}
│   ├── config.py
│   ├── data.py              # loading, cleaning, currency conversion, typology labels
│   ├── features.py          # leakage-safe behavioural and network features
│   ├── model.py             # time split, LightGBM, Isolation Forest
│   └── evaluate.py          # PR-AUC, precision/recall@k, per-typology recall
├── tests/                   # pytest suite, including a no-look-ahead test
├── data/raw/                # downloaded files (git-ignored)
├── data/processed/          # Parquet outputs (git-ignored)
├── models/                  # trained models (git-ignored)
└── reports/                 # metrics.json, summary.md, feature_importance.csv, top_alerts.csv
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Download `LI-Small_Trans.csv` and `LI-Small_Patterns.txt` into `data/raw/` (see
[data/raw/README.md](data/raw/README.md)). Then:

```bash
python -m aegis all       # ~12 minutes on a laptop; or run ingest / features / train separately
pytest
```

To use another dataset version, change `data.dataset` in `config.yaml` (e.g. `HI-Small`).

## Outputs

| File | Contents |
|---|---|
| `reports/summary.md` | Results table, recall by typology, top features |
| `reports/metrics.json` | All metrics and split details, machine-readable |
| `reports/top_alerts.csv` | The 1,000 highest-risk test transactions with accounts and scores, for a dashboard |
| `reports/feature_importance.csv` | LightGBM gain per feature |
| `models/LI-Small_lightgbm.txt` | Trained model |

## Data caveats

- **Sparse tail.** After 10 Sept the raw file has only 223 transactions, 61% of them
  laundering (attempts that run past the simulation end). Left in, they let a model learn
  "a quiet period means laundering" and inflate the test PR-AUC from 0.23 to 0.34. Ingest
  drops trailing days with under 1% of a median day's volume (`data.min_day_share`).
- **Typology coverage.** Only about a quarter of laundering transactions appear in the
  patterns file. Per-typology recall uses only those, which is why it is higher than overall recall.
- **Same-minute ordering.** Timestamps are per minute. Rolling windows can include other
  transactions from the same minute; look-ups of an account's previous inflow or outflow exclude them.

## Roadmap

- Graph features over a time window: cycle detection, PageRank, community membership
- Account-level risk scores and case grouping of related alerts
- SHAP explanations for each alert
- Graph neural network (GraphSAGE / GAT) comparison
- FastAPI scoring service and an investigator dashboard with a network view
- LLM-written alert summaries generated from the structured evidence only
