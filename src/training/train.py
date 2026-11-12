"""Stage 6 - Model Training.

Trains both models and tracks everything with MLflow:
  * parameters (hyperparameters, data sizes)
  * training metrics (MF train RMSE, loss curve)
  * artifacts (the pickled model files)

One MLflow run per model, under a shared experiment. The run_ids are returned so
the evaluation stage can log test-set metrics into the same runs, giving one
complete record (params + train + test) per model.
"""
from __future__ import annotations

import pickle

import mlflow
import pandas as pd

from src.training.models import MatrixFactorizationRecommender, PopularityRecommender
from src.utils import ensure_parent, get_logger, load_config, resolve

logger = get_logger("training")


def _setup_mlflow(cfg: dict) -> None:
    mlflow.set_tracking_uri(cfg["training"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["training"]["mlflow"]["experiment_name"])


def _save_pickle(obj, path: str) -> str:
    full = ensure_parent(resolve(path))
    with open(full, "wb") as fh:
        pickle.dump(obj, fh)
    return str(full)


def run(prep: dict | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if prep is None:
        from src.preparation import prepare
        prep = prepare.run(cfg=cfg)

    train = prep["train"]
    _setup_mlflow(cfg)
    run_ids: dict[str, str] = {}
    models: dict = {}

    # ---------------- Popularity baseline ---------------- #
    with mlflow.start_run(run_name="popularity_baseline") as run_pop:
        pop = PopularityRecommender().fit(train)
        mlflow.log_param("model_type", "popularity_baseline")
        mlflow.log_param("train_interactions", len(train))
        mlflow.log_param("train_users", train["user_id"].nunique())
        mlflow.log_param("train_items", train["item_id"].nunique())
        path = _save_pickle(pop, cfg["training"]["paths"]["popularity_model"])
        mlflow.log_artifact(path, artifact_path="model")
        run_ids["popularity"] = run_pop.info.run_id
        models["popularity"] = pop
        logger.info("Trained popularity baseline (run_id=%s)", run_pop.info.run_id)

    # ---------------- Matrix factorization ---------------- #
    mf_cfg = cfg["training"]["matrix_factorization"]
    with mlflow.start_run(run_name="matrix_factorization") as run_mf:
        mf = MatrixFactorizationRecommender(
            n_factors=mf_cfg["n_factors"],
            n_epochs=mf_cfg["n_epochs"],
            lr=mf_cfg["lr"],
            reg=mf_cfg["reg"],
            seed=cfg["project"]["random_seed"],
        ).fit(train)

        mlflow.log_param("model_type", "matrix_factorization")
        for key in ("n_factors", "n_epochs", "lr", "reg"):
            mlflow.log_param(key, mf_cfg[key])
        mlflow.log_param("train_interactions", len(train))
        mlflow.log_param("train_users", train["user_id"].nunique())
        mlflow.log_param("train_items", train["item_id"].nunique())
        # Log the training loss curve.
        for epoch, loss in enumerate(mf.train_loss_):
            mlflow.log_metric("train_rmse", loss, step=epoch)
        mlflow.log_metric("final_train_rmse", mf.train_loss_[-1])
        path = _save_pickle(mf, cfg["training"]["paths"]["mf_model"])
        mlflow.log_artifact(path, artifact_path="model")
        run_ids["mf"] = run_mf.info.run_id
        models["mf"] = mf
        logger.info(
            "Trained matrix factorization (run_id=%s, final_train_rmse=%.4f)",
            run_mf.info.run_id, mf.train_loss_[-1],
        )

    return {"models": models, "run_ids": run_ids}


if __name__ == "__main__":
    run()
