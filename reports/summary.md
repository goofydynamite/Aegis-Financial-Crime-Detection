# Results on LI-Small

Test period: 2022-09-08 16:06:00 to 2022-09-10 23:59:00, 1,384,828 transactions, 791 laundering.

| Model | PR-AUC | ROC-AUC | Precision@100 | Recall@100 | Precision@500 | Recall@500 | Precision@1000 | Recall@1000 | Precision@5000 | Recall@5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy_isolation_forest | 0.0005 | 0.4562 | 0.020 | 0.003 | 0.004 | 0.003 | 0.003 | 0.004 | 0.001 | 0.006 |
| isolation_forest | 0.0015 | 0.7777 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.009 |
| lightgbm | 0.2327 | 0.9658 | 0.950 | 0.120 | 0.336 | 0.212 | 0.205 | 0.259 | 0.071 | 0.448 |

## LightGBM recall by typology (top alerts)

| Typology | Transactions | Caught | Recall |
|---|---|---|---|
| BIPARTITE | 38 | 35 | 0.92 |
| CYCLE | 19 | 18 | 0.95 |
| FAN-IN | 7 | 6 | 0.86 |
| FAN-OUT | 28 | 25 | 0.89 |
| GATHER-SCATTER | 32 | 29 | 0.91 |
| RANDOM | 20 | 17 | 0.85 |
| SCATTER-GATHER | 46 | 35 | 0.76 |
| STACK | 19 | 12 | 0.63 |

## Top features (LightGBM gain)

| Feature | Share |
|---|---|
| r_secs_since_prev_in | 10.1% |
| amount_vs_sender_mean | 7.8% |
| payment_format_code | 7.1% |
| amount_vs_receiver_mean | 7.1% |
| s_secs_since_prev_out | 7.0% |
| r_in_sum_24h | 6.4% |
| log_amount_usd | 6.0% |
| s_last_in_amount_ratio | 5.8% |
| s_secs_since_last_in | 4.9% |
| s_distinct_receivers | 4.0% |
| s_out_prior_count | 4.0% |
| s_out_count_24h | 3.9% |
