"""Stage 3 + 4 - Data Preparation and Transformation.

Takes the validated (clean) tables and produces model-ready datasets:

  * final cleaning: impute residual missing values, standardise formats
  * time-based train/test split (no leakage: test = most recent interactions)
  * cold-start filtering on the TRAIN side only
  * derived datasets:
        - user_activity_summary  (per-user counts / averages / recency)
        - item_popularity        (per-item counts / averages / rank)
        - user_item_matrix        (pivot of train ratings, users x items)

The train/test split is strictly temporal: interactions before the time
cutoff (a quantile of the timestamp distribution) are training, the rest are
test. This mirrors how a recommender is used in production - predict the future
from the past - and avoids the optimistic bias of a random split.
"""
from __future__ import annotations

import pandas as pd

from src.utils import ensure_parent, get_logger, load_config, resolve

logger = get_logger("preparation")


def _clean_users(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    # Impute residual missing demographics with explicit sentinels.
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)
    df["gender"] = df["gender"].fillna("O")
    df["country"] = df["country"].fillna("UNKNOWN").str.upper().str.strip()
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.normalize()
    return df


def _clean_items(items: pd.DataFrame) -> pd.DataFrame:
    df = items.copy()
    df["category"] = df["category"].fillna("Unknown").str.title().str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["release_year"] = df["release_year"].fillna(df["release_year"].median()).astype(int)
    return df


def _clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("s")
    df["event_type"] = df["event_type"].str.lower().str.strip()
    # Deterministic ordering helps reproducibility of the split.
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _time_split(interactions: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoff = interactions["timestamp"].quantile(1 - test_fraction)
    train = interactions[interactions["timestamp"] < cutoff].copy()
    test = interactions[interactions["timestamp"] >= cutoff].copy()
    return train, test, cutoff


def _filter_cold_start(train: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iteratively drop users/items with too few TRAIN interactions."""
    df = train.copy()
    prev = -1
    while len(df) != prev:
        prev = len(df)
        uc = df["user_id"].value_counts()
        ic = df["item_id"].value_counts()
        keep_u = uc[uc >= min_user].index
        keep_i = ic[ic >= min_item].index
        df = df[df["user_id"].isin(keep_u) & df["item_id"].isin(keep_i)]
    return df.reset_index(drop=True)


def _user_activity_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    ref = interactions["timestamp"].max()
    g = interactions.groupby("user_id")
    out = pd.DataFrame(
        {
            "user_interaction_count": g.size(),
            "user_avg_rating": g["rating"].mean().round(4),
            "user_distinct_items": g["item_id"].nunique(),
            "user_first_interaction": g["timestamp"].min(),
            "user_last_interaction": g["timestamp"].max(),
        }
    ).reset_index()
    out["user_days_since_last"] = (ref - out["user_last_interaction"]).dt.days
    return out


def _item_popularity(interactions: pd.DataFrame) -> pd.DataFrame:
    g = interactions.groupby("item_id")
    out = pd.DataFrame(
        {
            "item_interaction_count": g.size(),
            "item_avg_rating": g["rating"].mean().round(4),
            "item_distinct_users": g["user_id"].nunique(),
        }
    ).reset_index()
    out["item_popularity_rank"] = (
        out["item_interaction_count"].rank(ascending=False, method="first").astype(int)
    )
    return out.sort_values("item_popularity_rank").reset_index(drop=True)


def _user_item_matrix(train: pd.DataFrame) -> pd.DataFrame:
    # Mean rating per (user, item) in case of repeat interactions.
    mat = train.pivot_table(
        index="user_id", columns="item_id", values="rating", aggfunc="mean"
    )
    return mat


def run(clean_tables: dict[str, pd.DataFrame] | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if clean_tables is None:
        from src.validation import validate
        clean_tables = validate.run(cfg=cfg)

    p = cfg["preparation"]
    users = _clean_users(clean_tables["users"])
    items = _clean_items(clean_tables["items"])
    interactions = _clean_interactions(clean_tables["interactions"])

    train, test, cutoff = _time_split(interactions, p["test_fraction"])
    logger.info(
        "Time split at %s -> train=%d test=%d", cutoff, len(train), len(test)
    )

    train = _filter_cold_start(train, p["min_user_interactions"], p["min_item_interactions"])
    # Restrict test to users/items seen in train (can't score unknown entities).
    train_users = set(train["user_id"].unique())
    train_items = set(train["item_id"].unique())
    test = test[test["user_id"].isin(train_users) & test["item_id"].isin(train_items)].reset_index(drop=True)
    logger.info(
        "After cold-start filter: train=%d (users=%d items=%d) | test=%d",
        len(train), len(train_users), len(train_items), len(test),
    )

    user_activity = _user_activity_summary(train)
    item_pop = _item_popularity(train)
    uim = _user_item_matrix(train)

    # Persist everything as parquet (columnar, typed, fast).
    out = {
        "users_clean": users,
        "items_clean": items,
        "interactions_clean": interactions,
        "train": train,
        "test": test,
        "user_activity": user_activity,
        "item_popularity": item_pop,
        "user_item_matrix": uim,
    }
    for key, path in p["paths"].items():
        out[key].to_parquet(ensure_parent(resolve(path)))
    logger.info("Processed datasets written to data/processed/")
    out["train_cutoff"] = cutoff
    return out


if __name__ == "__main__":
    run()
