"""Command-line entry point: python -m aegis {ingest,features,train,all}."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from . import data, evaluate, features, model
from .config import Config


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_ingest(cfg: Config) -> None:
    _log(f"Ingesting {cfg.raw_transactions.name}")
    df = data.ingest(cfg.raw_transactions, cfg.raw_patterns, cfg.transactions_parquet,
                     cfg["data"]["min_day_share"])
    _log(f"Wrote {len(df):,} rows ({df['is_laundering'].sum():,} laundering, "
         f"{df['pattern'].is_not_null().sum():,} with a typology) -> {cfg.transactions_parquet}")
    missing = df.filter(pl.col("amount_usd").is_null())["payment_currency"].unique().to_list()
    if missing:
        _log(f"WARNING: no USD rate for {missing}")


def run_features(cfg: Config) -> None:
    _log("Building features")
    cols = ["txn_id", "timestamp", "sender", "receiver", "amount_paid", "amount_usd",
            "payment_currency", "receiving_currency", "payment_format", "is_laundering", "pattern"]
    tx = pl.read_parquet(cfg.transactions_parquet, columns=cols)
    feats = features.build_features(tx)
    feats.write_parquet(cfg.features_parquet, compression="zstd")
    _log(f"Wrote {feats.width - 4} features for {len(feats):,} rows -> {cfg.features_parquet}")


def _score_report(name: str, y: np.ndarray, scores: np.ndarray, patterns: np.ndarray,
                  ev: dict) -> dict:
    metrics = evaluate.evaluate_scores(y, scores, ev["k_values"])
    metrics["pattern_recall"] = evaluate.pattern_recall(patterns, scores, ev["pattern_recall_k"])
    k = ev["k_values"]
    _log(f"{name:<24} PR-AUC {metrics['pr_auc']:.4f}  ROC-AUC {metrics['roc_auc']:.4f}  "
         + "  ".join(f"R@{x} {metrics[f'recall@{x}']:.3f}" for x in k))
    return metrics


def run_train(cfg: Config) -> None:
    ev = cfg["evaluation"]
    feats = pl.read_parquet(cfg.features_parquet)
    train, valid, test = model.time_split(feats, **cfg["split"])
    del feats
    split_info = {
        name: {"rows": len(part), "laundering": int(part["is_laundering"].sum()),
               "start": str(part["timestamp"].min()), "end": str(part["timestamp"].max())}
        for name, part in (("train", train), ("valid", valid), ("test", test))
    }
    for name, s in split_info.items():
        _log(f"{name:<5} {s['rows']:>9,} rows  {s['laundering']:>5,} laundering  {s['start']} -> {s['end']}")

    y_train = train["is_laundering"].to_numpy()
    y_valid = valid["is_laundering"].to_numpy()
    y_test = test["is_laundering"].to_numpy()
    patterns = test["pattern"].to_numpy()
    results: dict[str, dict] = {}

    _log("Baseline 1: Isolation Forest on the legacy prototype's features")
    X_tr, X_te = model.to_matrix(train, features.LEGACY_FEATURES), model.to_matrix(test, features.LEGACY_FEATURES)
    iso = model.train_isolation_forest(X_tr, seed=cfg.seed, **cfg["isolation_forest"])
    results["legacy_isolation_forest"] = _score_report(
        "legacy_isolation_forest", y_test, model.isolation_scores(iso, X_te), patterns, ev)

    X_tr, X_te = model.to_matrix(train, features.FEATURES), model.to_matrix(test, features.FEATURES)
    _log("Baseline 2: Isolation Forest on behavioural + network features")
    iso = model.train_isolation_forest(X_tr, seed=cfg.seed, **cfg["isolation_forest"])
    results["isolation_forest"] = _score_report(
        "isolation_forest", y_test, model.isolation_scores(iso, X_te), patterns, ev)
    del iso

    _log("Main model: LightGBM (early stopping on validation PR-AUC)")
    X_va = model.to_matrix(valid, features.FEATURES)
    booster = model.train_lightgbm(X_tr, y_train, X_va, y_valid, features.FEATURES,
                                   cfg["lightgbm"], cfg.seed)
    del X_tr, X_va
    test_scores = booster.predict(X_te, num_iteration=booster.best_iteration)
    results["lightgbm"] = _score_report("lightgbm", y_test, test_scores, patterns, ev)
    results["lightgbm"]["best_iteration"] = booster.best_iteration

    booster.save_model(str(cfg.models_dir / f"{cfg.dataset}_lightgbm.txt"),
                       num_iteration=booster.best_iteration)
    importance = pl.DataFrame({
        "feature": features.FEATURES,
        "gain": booster.feature_importance("gain", iteration=booster.best_iteration),
    }).sort("gain", descending=True).with_columns(share=pl.col("gain") / pl.col("gain").sum())
    importance.write_csv(cfg.reports_dir / "feature_importance.csv")

    _export_alerts(cfg, test, test_scores)
    report = {"dataset": cfg.dataset, "seed": cfg.seed, "splits": split_info,
              "features": features.FEATURES, "models": results}
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (cfg.reports_dir / "summary.md").write_text(_summary_markdown(report, importance), encoding="utf-8")
    _log(f"Reports written to {cfg.reports_dir}")


def _export_alerts(cfg: Config, test: pl.DataFrame, scores: np.ndarray) -> None:
    """Scored test set (for dashboards) and the top-k alert queue as CSV."""
    scored = test.select("txn_id", "is_laundering", "pattern").with_columns(
        risk_score=pl.Series(scores).cast(pl.Float32))
    tx = pl.read_parquet(cfg.transactions_parquet, columns=[
        "txn_id", "timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_paid", "payment_currency", "amount_usd", "payment_format"])
    scored = tx.join(scored, on="txn_id", how="inner").sort("risk_score", descending=True)
    scored.write_parquet(cfg.reports_dir / "scored_test.parquet", compression="zstd")
    scored.head(cfg["evaluation"]["top_alerts_export"]).with_columns(
        pl.col("payment_format").cast(pl.String)).write_csv(cfg.reports_dir / "top_alerts.csv")


def _summary_markdown(report: dict, importance: pl.DataFrame) -> str:
    models = report["models"]
    ks = [k for k in (100, 500, 1000, 5000) if f"recall@{k}" in models["lightgbm"]]
    lines = [f"# Results on {report['dataset']}", "",
             "Test period: {start} to {end}, {rows:,} transactions, {laundering:,} laundering.".format(
                 **report["splits"]["test"]), "",
             "| Model | PR-AUC | ROC-AUC | " + " | ".join(f"Precision@{k} | Recall@{k}" for k in ks) + " |",
             "|---|---|---|" + "---|---|" * len(ks)]
    for name, m in models.items():
        cells = " | ".join(f"{m[f'precision@{k}']:.3f} | {m[f'recall@{k}']:.3f}" for k in ks)
        lines.append(f"| {name} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {cells} |")
    pr = models["lightgbm"]["pattern_recall"]
    if pr:
        lines += ["", "## LightGBM recall by typology (top alerts)", "",
                  "| Typology | Transactions | Caught | Recall |", "|---|---|---|---|"]
        lines += [f"| {p} | {v['count']} | {v['caught']} | {v['recall']:.2f} |" for p, v in pr.items()]
    lines += ["", "## Top features (LightGBM gain)", "", "| Feature | Share |", "|---|---|"]
    lines += [f"| {r['feature']} | {r['share']:.1%} |" for r in importance.head(12).iter_rows(named=True)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="aegis", description=__doc__)
    parser.add_argument("command", choices=["ingest", "features", "train", "all"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    steps = {"ingest": [run_ingest], "features": [run_features], "train": [run_train],
             "all": [run_ingest, run_features, run_train]}[args.command]
    for step in steps:
        step(cfg)


if __name__ == "__main__":
    main()
