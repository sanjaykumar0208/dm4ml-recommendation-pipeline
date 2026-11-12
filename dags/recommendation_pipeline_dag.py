"""Apache Airflow DAG for the recommendation pipeline.

Each pipeline stage is an Airflow task. Dependencies are declared explicitly so
Airflow enforces the correct order and can retry / backfill individual stages:

    ingest >> validate >> prepare >> build_features >> train >> evaluate

The task callables re-use the exact same stage modules as the local
orchestrator (run_pipeline.py), so there is a single implementation of the
logic. Stages exchange data through the on-disk artifacts each stage persists
(data/, models/, reports/) rather than XCom, which keeps payloads small and
makes every stage independently inspectable and restartable.

Deployment:
    1. pip install apache-airflow
    2. Copy this file into $AIRFLOW_HOME/dags/ (or set dags_folder to ./dags).
    3. Make the project importable, e.g. add the repo root to PYTHONPATH, or set
       `export PYTHONPATH=/path/to/recommendation-pipeline:$PYTHONPATH`.
    4. airflow standalone   (then trigger `recommendation_pipeline` in the UI)
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.utils import load_config

# --------------------------------------------------------------------------- #
# Task callables - thin wrappers around the stage modules. Each reads the       #
# previous stage's persisted outputs from disk, so no XCom payloads are needed. #
# --------------------------------------------------------------------------- #
def task_ingest(**_):
    from src.ingestion import ingest
    ingest.run(load_config())


def task_validate(**_):
    import pandas as pd
    from src.utils import resolve
    from src.validation import validate
    cfg = load_config()
    p = cfg["ingestion"]["paths"]
    raw = {
        "users": pd.read_csv(resolve(p["users_raw"]), parse_dates=["signup_date"]),
        "items": pd.read_csv(resolve(p["items_raw"])),
        "interactions": pd.read_csv(resolve(p["interactions_raw"]), parse_dates=["timestamp"]),
    }
    validate.run(tables=raw, cfg=cfg)


def task_prepare(**_):
    # Preparation re-runs validation internally if clean tables aren't passed;
    # to avoid recompute we reload validated data via the validation stage which
    # reads the raw CSVs. Simpler: let prepare pull from validation.
    from src.preparation import prepare
    prepare.run(cfg=load_config())


def task_features(**_):
    from src.features import feature_store
    from src.preparation import prepare
    cfg = load_config()
    prep = _load_prep(cfg)
    feature_store.run(prep=prep, cfg=cfg)


def task_train(**_):
    from src.training import train
    cfg = load_config()
    prep = _load_prep(cfg)
    trained = train.run(prep=prep, cfg=cfg)
    # Persist run_ids so evaluate can attach test metrics to the same runs.
    import json
    from src.utils import ensure_parent, resolve
    with open(ensure_parent(resolve("reports/mlflow_run_ids.json")), "w") as fh:
        json.dump(trained["run_ids"], fh)


def task_evaluate(**_):
    import json
    import pickle
    from src.evaluation import evaluate
    from src.utils import resolve
    cfg = load_config()
    prep = _load_prep(cfg)
    with open(resolve(cfg["training"]["paths"]["popularity_model"]), "rb") as fh:
        pop = pickle.load(fh)
    with open(resolve(cfg["training"]["paths"]["mf_model"]), "rb") as fh:
        mf = pickle.load(fh)
    run_ids = {}
    try:
        with open(resolve("reports/mlflow_run_ids.json")) as fh:
            run_ids = json.load(fh)
    except FileNotFoundError:
        pass
    trained = {"models": {"popularity": pop, "mf": mf}, "run_ids": run_ids}
    evaluate.run(trained=trained, prep=prep, cfg=cfg)


def _load_prep(cfg) -> dict:
    """Reload the processed datasets a downstream stage needs from parquet."""
    import pandas as pd
    from src.utils import resolve
    p = cfg["preparation"]["paths"]
    prep = {k: pd.read_parquet(resolve(v)) for k, v in p.items()}
    return prep


# --------------------------------------------------------------------------- #
# DAG definition.                                                              #
# --------------------------------------------------------------------------- #
default_args = {
    "owner": "data-eng",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="recommendation_pipeline",
    description="End-to-end data management pipeline for a recommendation system",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",          # Airflow >= 2.4; use schedule_interval on older
    catchup=False,
    tags=["recommendation", "mlops", "dm4ml"],
) as dag:

    ingest = PythonOperator(task_id="ingest", python_callable=task_ingest)
    validate = PythonOperator(task_id="validate", python_callable=task_validate)
    prepare = PythonOperator(task_id="prepare_transform", python_callable=task_prepare)
    features = PythonOperator(task_id="build_features", python_callable=task_features)
    train = PythonOperator(task_id="train_models", python_callable=task_train)
    evaluate = PythonOperator(task_id="evaluate_models", python_callable=task_evaluate)

    ingest >> validate >> prepare >> features >> train >> evaluate
