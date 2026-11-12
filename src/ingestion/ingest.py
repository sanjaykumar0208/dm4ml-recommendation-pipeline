"""Stage 1 - Data Ingestion.

Produces three raw tables:
  * users         (user_id, age, gender, country, signup_date)
  * items         (item_id, title, category, release_year)
  * interactions  (user_id, item_id, rating, event_type, timestamp)

Two modes (see configs/config.yaml -> ingestion.mode):
  * synthetic : reproducible generator. Injects a configurable fraction of
                dirty records (missing values, out-of-range ratings, bad
                timestamps, duplicates, orphan foreign keys) so that the
                validation stage has genuine problems to detect.
  * csv       : read user-supplied CSVs already placed in data/raw/.

The generator mimics the MovieLens schema so real MovieLens data can be
swapped in with minimal column mapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ensure_parent, get_logger, load_config, resolve

logger = get_logger("ingestion")

_CATEGORIES = [
    "Action", "Comedy", "Drama", "Thriller", "Romance", "SciFi",
    "Horror", "Documentary", "Animation", "Fantasy", "Crime", "Adventure",
]
_COUNTRIES = ["IN", "US", "GB", "DE", "FR", "AU", "CA", "JP", "BR", "SG"]
_GENDERS = ["M", "F", "O"]
_EVENT_TYPES = ["view", "click", "rating", "purchase"]


def _generate_users(rng: np.random.Generator, n_users: int, cats: int) -> pd.DataFrame:
    user_ids = np.arange(1, n_users + 1)
    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "age": rng.integers(13, 80, size=n_users),
            "gender": rng.choice(_GENDERS, size=n_users, p=[0.48, 0.48, 0.04]),
            "country": rng.choice(_COUNTRIES, size=n_users),
            "signup_date": pd.to_datetime("2022-01-01")
            + pd.to_timedelta(rng.integers(0, 900, size=n_users), unit="D"),
        }
    )
    return df


def _generate_items(rng: np.random.Generator, n_items: int, cats: int) -> pd.DataFrame:
    item_ids = np.arange(1, n_items + 1)
    categories = _CATEGORIES[:cats]
    df = pd.DataFrame(
        {
            "item_id": item_ids,
            "title": [f"Movie_{i:04d}" for i in item_ids],
            "category": rng.choice(categories, size=n_items),
            "release_year": rng.integers(1980, 2025, size=n_items),
        }
    )
    return df


def _generate_interactions(
    rng: np.random.Generator,
    n_users: int,
    n_items: int,
    n_interactions: int,
    start_date: str,
    end_date: str,
    n_latent: int = 8,
) -> pd.DataFrame:
    """Generate interactions with *genuine latent preference structure*.

    Each user gets a latent taste vector and each item a latent attribute
    vector; their dot product is a real affinity signal that drives both which
    items a user interacts with and what rating they give. This is what makes
    the data realistic: a personalised model (matrix factorization) can recover
    the structure and beat the popularity baseline, instead of the ratings being
    pure noise. A lognormal popularity base keeps a realistic long tail so the
    popularity baseline is still a strong, non-trivial competitor.
    """
    # Latent factors (scaled so dot products stay in a sensible range).
    user_f = rng.normal(0, 1, size=(n_users, n_latent)) / np.sqrt(n_latent)
    item_f = rng.normal(0, 1, size=(n_items, n_latent)) / np.sqrt(n_latent)
    user_bias = rng.normal(0, 0.5, size=n_users)
    item_bias = rng.normal(0, 0.5, size=n_items)

    affinity = user_f @ item_f.T  # (n_users, n_items), real taste signal
    aff_std = (affinity - affinity.mean()) / (affinity.std() + 1e-9)

    # Per-user activity level (long tail of heavy vs light users).
    user_act = rng.lognormal(mean=0.0, sigma=0.6, size=n_users)
    user_act = user_act / user_act.sum()
    per_user = rng.multinomial(n_interactions, user_act)

    # Item choice is driven by *personal affinity*, not a shared global
    # popularity term. Item popularity therefore emerges from aggregate taste:
    # items that appeal to many users become popular. This keeps the popularity
    # baseline a genuine, non-trivial competitor while leaving real personalised
    # signal for collaborative filtering to exploit (gamma controls how sharply
    # taste concentrates each user's choices).
    gamma = 2.0
    start = pd.to_datetime(start_date).value
    end = pd.to_datetime(end_date).value

    u_list, i_list = [], []
    for u in range(n_users):
        n_u = per_user[u]
        if n_u == 0:
            continue
        logits = gamma * aff_std[u]
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        chosen = rng.choice(n_items, size=n_u, p=probs)
        u_list.append(np.full(n_u, u))
        i_list.append(chosen)

    u_idx = np.concatenate(u_list)
    i_idx = np.concatenate(i_list)
    n_rows = len(u_idx)

    # Rating = global mean + user bias + item bias + (strong) affinity + noise,
    # on the 0.5..5.0 half-star scale. Affinity is the dominant, low-noise signal
    # so a latent-factor model can recover it and rank well.
    raw = (
        3.4
        + 0.3 * user_bias[u_idx]
        + 0.3 * item_bias[i_idx]
        + 1.5 * aff_std[u_idx, i_idx]
        + rng.normal(0, 0.3, size=n_rows)
    )
    ratings = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)

    ts = pd.to_datetime(rng.integers(start, end, size=n_rows))
    event_types = rng.choice(_EVENT_TYPES, size=n_rows, p=[0.4, 0.3, 0.2, 0.1])

    df = pd.DataFrame(
        {
            "user_id": (u_idx + 1).astype(int),   # 1-based ids
            "item_id": (i_idx + 1).astype(int),
            "rating": ratings,
            "event_type": event_types,
            "timestamp": ts,
        }
    )
    return df


def _inject_dirt(
    rng: np.random.Generator,
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    dirty_fraction: float,
) -> pd.DataFrame:
    """Corrupt a fraction of interaction rows so validation has work to do."""
    df = interactions.copy()
    n = len(df)
    n_dirty = int(n * dirty_fraction)
    if n_dirty == 0:
        return df

    idx = rng.choice(df.index, size=n_dirty, replace=False)
    # Split the dirty rows across several defect types.
    buckets = np.array_split(idx, 5)

    # 1) Missing ratings.
    df.loc[buckets[0], "rating"] = np.nan
    # 2) Out-of-range ratings (negative / too high).
    df.loc[buckets[1], "rating"] = rng.choice([-2.0, 7.0, 99.0], size=len(buckets[1]))
    # 3) Invalid / future timestamps.
    df.loc[buckets[2], "timestamp"] = pd.Timestamp("2099-01-01")
    # 4) Orphan foreign keys (user_id / item_id that do not exist).
    df.loc[buckets[3], "user_id"] = n_users + rng.integers(1, 50, size=len(buckets[3]))
    df.loc[buckets[4], "item_id"] = n_items + rng.integers(1, 50, size=len(buckets[4]))

    # 5) Exact duplicates: re-append a sample of clean rows.
    dup = df.loc[rng.choice(df.index, size=max(1, n_dirty // 5), replace=False)]
    df = pd.concat([df, dup], ignore_index=True)

    # Shuffle so defects are not clustered at the end.
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return df


def generate_synthetic(cfg: dict) -> dict[str, pd.DataFrame]:
    s = cfg["ingestion"]["synthetic"]
    seed = cfg["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    logger.info("Generating synthetic data (seed=%d) ...", seed)
    users = _generate_users(rng, s["n_users"], s["n_categories"])
    items = _generate_items(rng, s["n_items"], s["n_categories"])
    interactions = _generate_interactions(
        rng, s["n_users"], s["n_items"], s["n_interactions"],
        s["start_date"], s["end_date"],
    )
    interactions = _inject_dirt(
        rng, interactions, s["n_users"], s["n_items"], s["dirty_fraction"]
    )
    logger.info(
        "Generated users=%d items=%d interactions=%d (incl. injected defects)",
        len(users), len(items), len(interactions),
    )
    return {"users": users, "items": items, "interactions": interactions}


def _read_csv(cfg: dict) -> dict[str, pd.DataFrame]:
    p = cfg["ingestion"]["paths"]
    logger.info("Reading raw CSVs from data/raw/ ...")
    users = pd.read_csv(resolve(p["users_raw"]), parse_dates=["signup_date"])
    items = pd.read_csv(resolve(p["items_raw"]))
    interactions = pd.read_csv(resolve(p["interactions_raw"]), parse_dates=["timestamp"])
    return {"users": users, "items": items, "interactions": interactions}


def run(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Entry point for the ingestion stage. Returns the three raw tables and
    also persists them to data/raw/ so the stage is independently inspectable."""
    cfg = cfg or load_config()
    mode = cfg["ingestion"]["mode"]

    if mode == "synthetic":
        tables = generate_synthetic(cfg)
    elif mode == "csv":
        tables = _read_csv(cfg)
    else:
        raise ValueError(f"Unknown ingestion mode: {mode!r}")

    paths = cfg["ingestion"]["paths"]
    tables["users"].to_csv(ensure_parent(resolve(paths["users_raw"])), index=False)
    tables["items"].to_csv(ensure_parent(resolve(paths["items_raw"])), index=False)
    tables["interactions"].to_csv(
        ensure_parent(resolve(paths["interactions_raw"])), index=False
    )
    logger.info("Raw tables written to data/raw/")
    return tables


if __name__ == "__main__":
    run()
