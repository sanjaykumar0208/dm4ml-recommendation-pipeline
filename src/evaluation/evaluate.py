"""Stage 7 - Model Evaluation.

Offline evaluation against the held-out (future) test split.

Ranking metrics (top-K, binary relevance = item appears in the user's test set):
  * Precision@K
  * Recall@K
  * NDCG@K
Catalogue / list metrics:
  * Coverage   - fraction of catalogue items that appear in any top-K list
  * Diversity  - mean intra-list category diversity (distinct categories / K)
Rating metrics (explicit-rating prediction):
  * RMSE, MAE  - over the test interactions

Both models are scored at every K in config, compared in a results table
(CSV + Markdown), and the test metrics are logged back into each model's
MLflow run.
"""
from __future__ import annotations

import math

import mlflow
import numpy as np
import pandas as pd

from src.utils import ensure_parent, get_logger, load_config, resolve

logger = get_logger("evaluation")


def _ndcg_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for pos, item in enumerate(recs[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(pos + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(p + 2) for p in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _ranking_metrics(model, test, train, item_cat: dict, catalog: set, k: int) -> dict:
    train_seen = train.groupby("user_id")["item_id"].agg(set).to_dict()
    test_relevant = test.groupby("user_id")["item_id"].agg(set).to_dict()
    users = list(test_relevant.keys())

    precisions, recalls, ndcgs, diversities = [], [], [], []
    recommended_items: set[int] = set()

    for u in users:
        relevant = test_relevant[u]
        seen = train_seen.get(u, set())
        recs = model.recommend(u, k, seen=seen)
        if not recs:
            continue
        hits = sum(1 for it in recs if it in relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant) if relevant else 0.0)
        ndcgs.append(_ndcg_at_k(recs, relevant, k))
        cats = [item_cat.get(it, "Unknown") for it in recs]
        diversities.append(len(set(cats)) / len(cats))
        recommended_items.update(recs)

    coverage = len(recommended_items) / len(catalog) if catalog else 0.0
    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"coverage@{k}": float(coverage),
        f"diversity@{k}": float(np.mean(diversities)) if diversities else 0.0,
    }


def _rating_metrics(model, test) -> dict:
    preds = np.array([model.predict(u, i) for u, i in zip(test["user_id"], test["item_id"])])
    actual = test["rating"].to_numpy(dtype=float)
    err = preds - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": rmse, "mae": mae}


def _evaluate_model(model, prep: dict, item_cat: dict, catalog: set, k_values: list[int]) -> dict:
    test, train = prep["test"], prep["train"]
    result: dict = {}
    for k in k_values:
        result.update(_ranking_metrics(model, test, train, item_cat, catalog, k))
    result.update(_rating_metrics(model, test))
    return result


def run(trained: dict | None = None, prep: dict | None = None, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    if prep is None:
        from src.preparation import prepare
        prep = prepare.run(cfg=cfg)
    if trained is None:
        from src.training import train
        trained = train.run(prep=prep, cfg=cfg)

    models = trained["models"]
    run_ids = trained.get("run_ids", {})
    k_values = cfg["evaluation"]["k_values"]

    items = prep["items_clean"]
    item_cat = dict(zip(items["item_id"], items["category"]))
    catalog = set(prep["train"]["item_id"].unique())

    rows = []
    name_map = {"popularity": "Popularity Baseline", "mf": "Matrix Factorization"}
    for key, model in models.items():
        logger.info("Evaluating %s ...", name_map.get(key, key))
        metrics = _evaluate_model(model, prep, item_cat, catalog, k_values)
        metrics_row = {"model": name_map.get(key, key), **{m: round(v, 4) for m, v in metrics.items()}}
        rows.append(metrics_row)

        # Log test metrics into the model's MLflow run.
        run_id = run_ids.get(key)
        if run_id:
            mlflow.set_tracking_uri(cfg["training"]["mlflow"]["tracking_uri"])
            with mlflow.start_run(run_id=run_id):
                for m, v in metrics.items():
                    mlflow.log_metric("test_" + m.replace("@", "_at_"), v)

    results = pd.DataFrame(rows)
    results.to_csv(ensure_parent(resolve(cfg["evaluation"]["results_path"])), index=False)
    with open(ensure_parent(resolve(cfg["evaluation"]["results_md"])), "w") as fh:
        fh.write("# Model Evaluation Results\n\n")
        fh.write(results.to_markdown(index=False))
        fh.write("\n")
    logger.info("Evaluation results written to reports/")
    print("\n" + results.to_markdown(index=False) + "\n")
    return results


if __name__ == "__main__":
    run()
