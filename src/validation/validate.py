"""Stage 2 - Data Validation.

Automated data-quality checks on the three raw tables:

  * schema / type checks         (pandera)
  * missing (null) values        (pandera nullable=False)
  * value-range checks           (rating in [0.5, 5.0], age plausible, year)
  * allowed-value / enum checks  (event_type, gender)
  * invalid timestamps           (not null, not in the future)
  * duplicate records            (pandas)
  * referential integrity        (interaction FKs must resolve to users/items)

Every failing interaction row is tagged with a human-readable reason and routed
to data/errors/. Clean rows pass through. A JSON report summarising every check
is written to reports/validation_report.json.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.utils import ensure_parent, get_logger, load_config, resolve

logger = get_logger("validation")


# --------------------------------------------------------------------------- #
# Pandera schemas (type + null + range + enum checks).                         #
# `coerce=True` casts where possible; failures surface as schema errors.       #
# --------------------------------------------------------------------------- #
def _user_schema() -> DataFrameSchema:
    return DataFrameSchema(
        {
            "user_id": Column(int, nullable=False, unique=True),
            "age": Column(int, Check.in_range(5, 120), nullable=True),
            "gender": Column(str, Check.isin(["M", "F", "O"]), nullable=True),
            "country": Column(str, nullable=True),
            "signup_date": Column("datetime64[ns]", nullable=True),
        },
        strict=False,
        coerce=True,
    )


def _item_schema() -> DataFrameSchema:
    return DataFrameSchema(
        {
            "item_id": Column(int, nullable=False, unique=True),
            "title": Column(str, nullable=False),
            "category": Column(str, nullable=False),
            "release_year": Column(int, Check.in_range(1900, 2030), nullable=True),
        },
        strict=False,
        coerce=True,
    )


def _interaction_schema(cfg: dict) -> DataFrameSchema:
    v = cfg["validation"]
    return DataFrameSchema(
        {
            "user_id": Column(int, nullable=False),
            "item_id": Column(int, nullable=False),
            "rating": Column(
                float,
                Check.in_range(v["rating_min"], v["rating_max"]),
                nullable=False,
            ),
            "event_type": Column(str, Check.isin(v["valid_event_types"]), nullable=False),
            "timestamp": Column("datetime64[ns]", nullable=False),
        },
        strict=False,
        coerce=True,
    )


def _failing_indices(df: pd.DataFrame, schema: DataFrameSchema) -> tuple[set, pd.DataFrame]:
    """Run a pandera schema lazily and return (failing_index_set, failure_cases)."""
    try:
        schema.validate(df, lazy=True)
        return set(), pd.DataFrame()
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        # `index` column holds the original DataFrame index of each failure.
        bad = cases["index"].dropna().unique().tolist()
        return set(bad), cases


def _validate_interactions(
    interactions: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = interactions.reset_index(drop=True).copy()
    reasons = pd.Series([""] * len(df), index=df.index)
    stats: dict[str, int] = {}

    # --- 1. schema / type / range / enum / null via pandera ----------------- #
    schema = _interaction_schema(cfg)
    schema_bad, cases = _failing_indices(df, schema)
    if not cases.empty:
        # Attribute each failure to its column/check for a readable reason.
        for _, row in cases.iterrows():
            i = row["index"]
            if pd.isna(i):
                continue
            col = row.get("column") or "schema"
            reasons.loc[i] = (reasons.loc[i] + f"{col}_invalid;").lstrip()
    stats["schema_type_range_enum_null_failures"] = len(schema_bad)

    # --- 2. explicit future-timestamp check --------------------------------- #
    now = pd.Timestamp(datetime.now())
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    future_mask = ts.notna() & (ts > now)
    for i in df.index[future_mask]:
        reasons.loc[i] += "timestamp_in_future;"
    stats["future_timestamps"] = int(future_mask.sum())

    # --- 3. referential integrity (FK) -------------------------------------- #
    valid_users = set(users["user_id"].dropna().astype("int64"))
    valid_items = set(items["item_id"].dropna().astype("int64"))
    uid = pd.to_numeric(df["user_id"], errors="coerce")
    iid = pd.to_numeric(df["item_id"], errors="coerce")
    orphan_user = ~uid.isin(valid_users) | uid.isna()
    orphan_item = ~iid.isin(valid_items) | iid.isna()
    for i in df.index[orphan_user]:
        reasons.loc[i] += "orphan_user_id;"
    for i in df.index[orphan_item]:
        reasons.loc[i] += "orphan_item_id;"
    stats["orphan_user_fk"] = int(orphan_user.sum())
    stats["orphan_item_fk"] = int(orphan_item.sum())

    # --- 4. duplicates (keep first, flag the rest) -------------------------- #
    dup_mask = df.duplicated(
        subset=["user_id", "item_id", "timestamp", "event_type"], keep="first"
    )
    for i in df.index[dup_mask]:
        reasons.loc[i] += "duplicate_record;"
    stats["duplicate_records"] = int(dup_mask.sum())

    # --- assemble clean / error partitions ---------------------------------- #
    bad_mask = reasons.str.len() > 0
    error_df = df[bad_mask].copy()
    error_df["error_reasons"] = reasons[bad_mask].str.rstrip(";")
    clean_df = df[~bad_mask].copy()

    # Coerce clean dtypes for downstream stages.
    clean_df["user_id"] = clean_df["user_id"].astype("int64")
    clean_df["item_id"] = clean_df["item_id"].astype("int64")
    clean_df["rating"] = clean_df["rating"].astype("float64")
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

    stats["total_input"] = len(df)
    stats["total_clean"] = len(clean_df)
    stats["total_errors"] = len(error_df)
    return clean_df, error_df, stats


def _validate_table(df: pd.DataFrame, schema: DataFrameSchema, key: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df.reset_index(drop=True).copy()
    bad, cases = _failing_indices(df, schema)
    reasons = pd.Series([""] * len(df), index=df.index)
    if not cases.empty:
        for _, row in cases.iterrows():
            i = row["index"]
            if pd.isna(i):
                continue
            col = row.get("column") or "schema"
            reasons.loc[i] += f"{col}_invalid;"
    # Duplicate primary keys.
    dup_mask = df.duplicated(subset=[key], keep="first")
    for i in df.index[dup_mask]:
        reasons.loc[i] += "duplicate_pk;"
    bad_mask = reasons.str.len() > 0
    error_df = df[bad_mask].copy()
    error_df["error_reasons"] = reasons[bad_mask].str.rstrip(";")
    clean_df = df[~bad_mask].copy()
    stats = {
        "total_input": len(df),
        "total_clean": len(clean_df),
        "total_errors": len(error_df),
        "duplicate_pk": int(dup_mask.sum()),
    }
    return clean_df, error_df, stats


def run(tables: dict[str, pd.DataFrame] | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if tables is None:
        from src.ingestion import ingest
        tables = ingest.run(cfg)

    logger.info("Validating users / items / interactions ...")
    users_clean, users_err, users_stats = _validate_table(tables["users"], _user_schema(), "user_id")
    items_clean, items_err, items_stats = _validate_table(tables["items"], _item_schema(), "item_id")
    inter_clean, inter_err, inter_stats = _validate_interactions(
        tables["interactions"], users_clean, items_clean, cfg
    )

    # Persist error datasets.
    ep = cfg["validation"]["error_paths"]
    users_err.to_csv(ensure_parent(resolve(ep["users"])), index=False)
    items_err.to_csv(ensure_parent(resolve(ep["items"])), index=False)
    inter_err.to_csv(ensure_parent(resolve(ep["interactions"])), index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "users": users_stats,
        "items": items_stats,
        "interactions": inter_stats,
    }
    with open(ensure_parent(resolve(cfg["validation"]["report_path"])), "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info(
        "Validation done. interactions: %d clean / %d errors (%.1f%% rejected).",
        inter_stats["total_clean"], inter_stats["total_errors"],
        100 * inter_stats["total_errors"] / max(inter_stats["total_input"], 1),
    )
    logger.info("Validation report -> %s", cfg["validation"]["report_path"])

    return {
        "users": users_clean,
        "items": items_clean,
        "interactions": inter_clean,
        "report": report,
    }


if __name__ == "__main__":
    run()
