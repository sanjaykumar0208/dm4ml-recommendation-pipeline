"""End-to-end pipeline orchestrator (plain Python workflow).

Runs every stage in dependency order, passing in-memory artifacts between
stages for speed while each stage also persists its outputs to disk so any
stage can be re-run or inspected independently.

    Ingestion -> Validation -> Preparation/Transformation -> Features
              -> Training (MLflow) -> Evaluation

Usage:
    python run_pipeline.py                 # full run
    python run_pipeline.py --config path   # custom config

The same stage functions are called by the Airflow DAG in dags/, so the two
orchestrators share identical logic (no drift between local and scheduled runs).
"""
from __future__ import annotations

import argparse
import time

from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.utils import get_logger, load_config
from src.validation import validate

logger = get_logger("orchestrator")


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STARTING END-TO-END RECOMMENDATION PIPELINE")
    logger.info("=" * 70)

    raw = ingest.run(cfg)
    clean = validate.run(tables=raw, cfg=cfg)
    prep = prepare.run(clean_tables=clean, cfg=cfg)
    feature_store.run(prep=prep, cfg=cfg)
    trained = train.run(prep=prep, cfg=cfg)
    evaluate.run(trained=trained, prep=prep, cfg=cfg)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE in %.1fs", time.time() - t0)
    logger.info("Artifacts: data/  models/  reports/  mlruns/  logs/")
    logger.info("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the recommendation pipeline.")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    args = ap.parse_args()
    main(args.config)
