# End-to-End Data Management Pipeline for a Recommendation System

**DM4ML – Assignment 2**

A modular, reproducible data-management pipeline for a **movie recommendation**
system. It ingests raw data, validates its quality, prepares and transforms it,
builds a feature store, trains and evaluates two recommender models, and runs the
whole thing through an orchestrator.

```
Ingestion → Validation → Preparation/Transformation → Feature Management
          → Model Training (MLflow) → Model Evaluation → Orchestration
```

The pipeline uses **reproducible synthetic data** by default (no external
downloads, fully deterministic via a fixed seed). The synthetic generator mimics
the MovieLens schema, so real MovieLens data can be dropped in with minimal
column mapping (see *Using real data* below).

---

## 1. Setup

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the full pipeline

The simplest way — the plain Python orchestrator (no Airflow required):

```bash
python run_pipeline.py
```

This runs every stage in order and writes all artifacts to `data/`, `models/`,
`reports/`, `mlflow.db` and `logs/`. A complete run takes ~10 seconds.

You can also run any stage on its own (each persists its outputs and pulls what
it needs from the previous stage):

```bash
python -m src.ingestion.ingest
python -m src.validation.validate
python -m src.preparation.prepare
python -m src.features.feature_store
python -m src.training.train
python -m src.evaluation.evaluate
```

## 3. View experiment tracking (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000  → experiment "movie-recommendation"
```

Each model is one run, logging hyperparameters, the MF training-loss curve, all
test metrics (`test_precision_at_10`, `test_rmse`, …) and the pickled model as an
artifact.

## 4. Run via Apache Airflow (orchestration path)

```bash
pip install "apache-airflow>=2.7,<3.0"
export AIRFLOW_HOME="$PWD/.airflow"
export PYTHONPATH="$PWD:$PYTHONPATH"          # make `src` importable from the DAG
# point Airflow at this repo's dags/ folder, or copy the DAG into $AIRFLOW_HOME/dags/
airflow standalone
# In the UI, enable + trigger the `recommendation_pipeline` DAG, then screenshot
# the Graph view for the deliverable.
```

The DAG (`dags/recommendation_pipeline_dag.py`) calls the **same stage modules**
as `run_pipeline.py`, so local and scheduled runs share one implementation.
A pre-rendered successful-run graph is in `reports/dag_run.png`, and the console
log of a real end-to-end run is in `reports/successful_run_log.txt`.

---

## Project structure

```
recommendation-pipeline/
├── configs/config.yaml         # single source of truth for all settings
├── run_pipeline.py             # plain-Python orchestrator
├── src/
│   ├── utils.py                # config loader, logger, IO helpers
│   ├── ingestion/ingest.py     # Stage 1: synthetic gen / CSV read
│   ├── validation/validate.py  # Stage 2: pandera + integrity checks
│   ├── preparation/prepare.py  # Stage 3+4: clean, time-split, derived data
│   ├── features/feature_store.py # Stage 5: feature store + catalog
│   ├── training/
│   │   ├── models.py           # PopularityRecommender, MatrixFactorization
│   │   └── train.py            # Stage 6: fit + MLflow tracking
│   └── evaluation/evaluate.py  # Stage 7: ranking + rating metrics
├── dags/recommendation_pipeline_dag.py   # Airflow DAG
├── data/{raw,processed,features,errors}/ # pipeline artifacts
├── models/                     # pickled trained models
├── reports/                    # validation report, feature catalog, results,
│                               #   architecture + DAG diagrams, run log
└── logs/pipeline.log
```

---

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|--------------|
| **1. Ingestion** | `src/ingestion` | Generates reproducible synthetic users/items/interactions (or reads CSVs). Injects a configurable fraction of dirty rows so validation has real defects to catch. |
| **2. Validation** | `src/validation` | Pandera schemas (types, nulls, ranges, enums) + explicit checks for future timestamps, duplicates and **referential integrity**. Bad rows are tagged with a reason and routed to `data/errors/`; a JSON summary goes to `reports/validation_report.json`. |
| **3+4. Prepare / Transform** | `src/preparation` | Cleans and imputes, standardises formats, does a **time-based** train/test split (no leakage), filters cold-start users/items, and builds derived datasets: user-activity summary, item popularity, and the user-item matrix. |
| **5. Feature management** | `src/features` | A file-based feature store computing user, item and interaction features, served through one lookup API used identically for training and prediction (no train/serve skew). Emits a documented `feature_catalog.csv`. |
| **6. Training** | `src/training` | Trains a **popularity baseline** and a **matrix-factorization** model (FunkSVD with biases, SGD). All params/metrics/artifacts tracked in MLflow. |
| **7. Evaluation** | `src/evaluation` | Precision@K, Recall@K, NDCG@K, Coverage, Diversity, plus RMSE/MAE for rating prediction. Writes a comparison table (`reports/evaluation_results.{csv,md}`). |
| **Orchestration** | `run_pipeline.py`, `dags/` | Two interchangeable orchestrators over the same stage functions. |

---

## Results (default synthetic run, K=10)

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 | RMSE | MAE |
|-------|-------------:|----------:|--------:|------------:|-------------:|-----:|----:|
| Popularity Baseline  | 0.0186 | 0.0346 | 0.0279 | 0.0633 | 0.878 | 0.536 | 0.340 |
| Matrix Factorization | 0.0200 | 0.0338 | 0.0284 | 0.4133 | 0.793 | 0.491 | 0.301 |

Matrix factorization clearly wins on **rating accuracy** (RMSE/MAE) and
**catalogue coverage**, and edges out the baseline on top-10 ranking, while the
popularity baseline remains a strong competitor on raw top-rank hit-rate — a
realistic, well-documented trade-off discussed in the final report.

*(Numbers are reproducible with the committed seed; re-running overwrites the
`reports/` outputs with identical values.)*

---

## Using real data instead of synthetic

1. Download MovieLens (e.g. `ml-latest-small`).
2. Produce three CSVs matching the schema and place them in `data/raw/`:
   - `users.csv`        → `user_id, age, gender, country, signup_date`
   - `items.csv`        → `item_id, title, category, release_year`
   - `interactions.csv` → `user_id, item_id, rating, event_type, timestamp`
3. Set `ingestion.mode: csv` in `configs/config.yaml` and re-run.

---

## Configuration

Everything is driven by `configs/config.yaml`: data sizes, dirty-data fraction,
validation thresholds, split fraction, cold-start cutoffs, MF hyperparameters,
K values, and all output paths. Change settings there — no code edits needed.
