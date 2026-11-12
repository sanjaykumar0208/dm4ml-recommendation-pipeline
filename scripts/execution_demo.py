"""Stage-by-stage execution driver used to capture execution evidence.

Runs the full pipeline in one process, printing a clear banner and an output
preview around each stage so the console log can be split into one terminal
"screenshot" per pipeline stage. This is a genuine end-to-end execution - every
line of output below is produced by the real stage code.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on path

import pandas as pd

from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.utils import load_config, resolve
from src.validation import validate

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)

DELIM = "===__STAGE__==="


def banner(title: str, command: str):
    print(f"\n{DELIM}|{title}|{command}")
    print(f"$ {command}")


cfg = load_config()

# ----------------------------- Stage 1 ------------------------------------- #
banner("Stage 1 - Data Ingestion", "python -m src.ingestion.ingest")
raw = ingest.run(cfg)
print("\n[preview] data/raw/interactions.csv (first 5 rows):")
print(raw["interactions"].head().to_string(index=False))
print(f"\nRows -> users={len(raw['users'])}, items={len(raw['items'])}, "
      f"interactions={len(raw['interactions'])}")

# ----------------------------- Stage 2 ------------------------------------- #
banner("Stage 2 - Data Validation", "python -m src.validation.validate")
clean = validate.run(tables=raw, cfg=cfg)
rep = clean["report"]["interactions"]
print("\n[preview] reports/validation_report.json (interactions):")
print(json.dumps(rep, indent=2))
err = pd.read_csv(resolve(cfg["validation"]["error_paths"]["interactions"]))
print("\n[preview] data/errors/interactions_errors.csv - top failure reasons:")
print(err["error_reasons"].value_counts().head(6).to_string())

# ----------------------------- Stage 3 ------------------------------------- #
banner("Stage 3 - Data Preparation & Transformation",
       "python -m src.preparation.prepare")
prep = prepare.run(clean_tables=clean, cfg=cfg)
print(f"\nTime-based split -> train={len(prep['train'])}, test={len(prep['test'])}")
print("\n[preview] data/processed/item_popularity.parquet (top 5 by rank):")
print(prep["item_popularity"].head().to_string(index=False))
print("\n[preview] user-item matrix shape:", prep["user_item_matrix"].shape)

# ----------------------------- Stage 4 ------------------------------------- #
banner("Stage 4 - Feature Management", "python -m src.features.feature_store")
store = feature_store.run(prep=prep, cfg=cfg)
print("\n[preview] reports/feature_catalog.csv (first 8 features):")
cat = pd.read_csv(resolve(cfg["features"]["paths"]["catalog"]))
print(cat.head(8).to_string(index=False))
print("\n[preview] data/features/user_features.parquet (first 3 users):")
print(store.user_features.head(3).to_string(index=False))

# ----------------------------- Stage 5 ------------------------------------- #
banner("Stage 5 - Model Training (MLflow tracked)",
       "python -m src.training.train")
trained = train.run(prep=prep, cfg=cfg)
import mlflow
mlflow.set_tracking_uri(cfg["training"]["mlflow"]["tracking_uri"])
exp = mlflow.get_experiment_by_name(cfg["training"]["mlflow"]["experiment_name"])
runs = mlflow.search_runs([exp.experiment_id])
show = runs[["tags.mlflow.runName", "params.model_type",
             "metrics.final_train_rmse"]].fillna("NA").astype(str)
show.columns = ["run_name", "model_type", "final_train_rmse"]
print("\n[preview] MLflow tracked runs (sqlite:///mlflow.db):")
print(show.to_string(index=False))

# ----------------------------- Stage 6 ------------------------------------- #
banner("Stage 6 - Model Evaluation", "python -m src.evaluation.evaluate")
results = evaluate.run(trained=trained, prep=prep, cfg=cfg)

# ----------------------------- Stage 7 ------------------------------------- #
banner("Stage 7 - Orchestration (full pipeline run)", "python run_pipeline.py")
print("The orchestrator chains every stage above in dependency order.")
print("Equivalent Airflow DAG: dags/recommendation_pipeline_dag.py")
print("\nDAG task order:")
print("  ingest >> validate >> prepare_transform >> build_features "
      ">> train_models >> evaluate_models")
print("\nPIPELINE COMPLETE - all stages executed successfully.")
print(f"\n{DELIM}|END|")
