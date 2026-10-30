from datetime import datetime, timedelta

import polars as pl
import pytest

from aegis.data import PAYMENT_FORMAT_DTYPE
from aegis.features import FEATURES, build_features

T0 = datetime(2022, 9, 1, 8, 0)


def make_tx(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    """rows: (minutes after T0, sender, receiver, amount_usd)."""
    return pl.DataFrame({
        "txn_id": list(range(len(rows))),
        "timestamp": [T0 + timedelta(minutes=m) for m, *_ in rows],
        "sender": [s for _, s, _, _ in rows],
        "receiver": [r for _, _, r, _ in rows],
        "amount_paid": [a for *_, a in rows],
        "amount_usd": [a for *_, a in rows],
        "payment_currency": ["US Dollar"] * len(rows),
        "receiving_currency": ["US Dollar"] * len(rows),
        "payment_format": pl.Series(["ACH"] * len(rows), dtype=PAYMENT_FORMAT_DTYPE),
        "is_laundering": [0] * len(rows),
        "pattern": pl.Series([None] * len(rows), dtype=pl.String),
    }).with_columns(pl.col("sender", "receiver").cast(pl.UInt32))


def test_sender_history_counts_only_earlier_payments():
    f = build_features(make_tx([(0, 1, 2, 100.0), (10, 1, 3, 300.0), (20, 1, 2, 50.0)]))
    assert f["s_out_prior_count"].to_list() == [0, 1, 2]
    assert f["s_distinct_receivers"].to_list() == [1, 2, 2]
    assert f["s_is_new_receiver"].to_list() == [1, 1, 0]
    assert f["amount_vs_sender_mean"][2] == pytest.approx(50.0 / 200.0)
    assert f["amount_vs_sender_mean"][0] is None


def test_24h_window_drops_old_payments():
    f = build_features(make_tx([(0, 1, 2, 100.0), (60 * 25, 1, 3, 10.0)]))
    assert f["s_out_count_24h"].to_list() == [1, 1]
    assert f["s_out_sum_24h"][1] == pytest.approx(10.0)


def test_pass_through_is_visible_from_sender_side():
    # account 2 receives 1000 and forwards 990 five minutes later
    f = build_features(make_tx([(0, 1, 2, 1000.0), (5, 2, 3, 990.0)]))
    assert f["s_secs_since_last_in"][1] == pytest.approx(300.0)
    assert f["s_last_in_amount_ratio"][1] == pytest.approx(0.99)
    assert f["s_secs_since_last_in"][0] is None


def test_fan_in_counts_distinct_senders():
    f = build_features(make_tx([(i, 10 + i, 99, 500.0) for i in range(5)] + [(9, 10, 99, 1.0)]))
    assert f["r_distinct_senders"].to_list() == [1, 2, 3, 4, 5, 5]


def test_features_do_not_look_ahead():
    base = [(0, 1, 2, 100.0), (30, 2, 3, 90.0), (60, 1, 3, 40.0)]
    future = [(120, 3, 1, 5000.0), (130, 2, 1, 7000.0), (140, 1, 2, 1.0)]
    a = build_features(make_tx(base)).select(FEATURES)
    b = build_features(make_tx(base + future)).select(FEATURES).head(len(base))
    assert a.equals(b)
