"""Stage 5 - Feature Management.

A lightweight, file-based feature layer. Features are computed once from the
TRAIN split, persisted as parquet, and served through a small `FeatureStore`
class whose lookup methods are used identically at training time and at
prediction/serving time - which is the whole point of a feature store
(train/serve consistency, no feature skew).

Three feature groups:
  * user features        (behavioural + demographic)
  * item features        (popularity + content)
  * interaction features (documented in the catalog; computed on the fly from
                          the two groups above, e.g. user-category affinity)

A machine-readable feature catalog (reports/feature_catalog.csv) documents every
feature: name, entity, dtype, description and source stage.
"""
from __future__ import annotations

import pandas as pd

from src.utils import ensure_parent, get_logger, load_config, resolve

logger = get_logger("features")


# --------------------------------------------------------------------------- #
# Feature catalog - the single source of truth for what each feature means.    #
# --------------------------------------------------------------------------- #
FEATURE_CATALOG = [
    # user features
    ("user_interaction_count", "user", "int", "Number of training interactions by the user", "activity_summary"),
    ("user_avg_rating", "user", "float", "Mean rating the user gives", "activity_summary"),
    ("user_rating_std", "user", "float", "Std dev of the user's ratings (rating pickiness)", "features"),
    ("user_distinct_items", "user", "int", "Count of distinct items the user interacted with", "activity_summary"),
    ("user_days_since_last", "user", "int", "Days from the user's last interaction to the data reference date", "activity_summary"),
    ("user_preferred_category", "user", "str", "Category the user interacts with most (mode)", "features"),
    ("user_age", "user", "int", "User age (demographic)", "users_clean"),
    ("user_country", "user", "str", "User country (demographic)", "users_clean"),
    # item features
    ("item_interaction_count", "item", "int", "Number of training interactions on the item (popularity)", "item_popularity"),
    ("item_avg_rating", "item", "float", "Mean rating received by the item", "item_popularity"),
    ("item_rating_std", "item", "float", "Std dev of the item's ratings (rating consistency)", "features"),
    ("item_distinct_users", "item", "int", "Count of distinct users who interacted with the item", "item_popularity"),
    ("item_popularity_rank", "item", "int", "Dense rank of the item by interaction count (1 = most popular)", "item_popularity"),
    ("item_category", "item", "str", "Item category / genre (content)", "items_clean"),
    ("item_age_years", "item", "int", "Years since the item's release year", "features"),
    # interaction features (computed at serve time)
    ("user_category_affinity", "interaction", "float", "Fraction of a user's interactions that fall in a given item's category", "serving"),
    ("rating_vs_item_avg", "interaction", "float", "A rating minus the item's average rating (bias signal)", "serving"),
]


class FeatureStore:
    """File-based feature store. Build once from train data; look up anywhere."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.user_features: pd.DataFrame | None = None
        self.item_features: pd.DataFrame | None = None

    # ----- build (offline, from train) ------------------------------------- #
    def build(self, prep: dict) -> "FeatureStore":
        train = prep["train"]
        users = prep["users_clean"]
        items = prep["items_clean"]
        activity = prep["user_activity"]
        item_pop = prep["item_popularity"]

        # --- user features --- #
        rating_std = train.groupby("user_id")["rating"].std().rename("user_rating_std")
        # Preferred category = the category the user interacts with most.
        cat = train.merge(items[["item_id", "category"]], on="item_id", how="left")
        pref_cat = (
            cat.groupby("user_id")["category"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
            .rename("user_preferred_category")
        )
        uf = (
            activity.merge(rating_std, on="user_id", how="left")
            .merge(pref_cat, on="user_id", how="left")
            .merge(users[["user_id", "age", "country"]], on="user_id", how="left")
            .rename(columns={"age": "user_age", "country": "user_country"})
        )
        uf["user_rating_std"] = uf["user_rating_std"].fillna(0.0).round(4)

        # --- item features --- #
        i_rating_std = train.groupby("item_id")["rating"].std().rename("item_rating_std")
        ref_year = pd.Timestamp.now().year
        itf = (
            item_pop.merge(i_rating_std, on="item_id", how="left")
            .merge(items[["item_id", "category", "release_year"]], on="item_id", how="left")
            .rename(columns={"category": "item_category"})
        )
        itf["item_rating_std"] = itf["item_rating_std"].fillna(0.0).round(4)
        itf["item_age_years"] = (ref_year - itf["release_year"]).clip(lower=0)

        self.user_features = uf
        self.item_features = itf
        return self

    # ----- persist / load -------------------------------------------------- #
    def save(self) -> None:
        fp = self.cfg["features"]["paths"]
        self.user_features.to_parquet(ensure_parent(resolve(fp["user_features"])))
        self.item_features.to_parquet(ensure_parent(resolve(fp["item_features"])))
        catalog = pd.DataFrame(
            FEATURE_CATALOG,
            columns=["feature_name", "entity", "dtype", "description", "source_stage"],
        )
        catalog.to_csv(ensure_parent(resolve(fp["catalog"])), index=False)
        logger.info("User/item features + feature catalog persisted.")

    def load(self) -> "FeatureStore":
        fp = self.cfg["features"]["paths"]
        self.user_features = pd.read_parquet(resolve(fp["user_features"]))
        self.item_features = pd.read_parquet(resolve(fp["item_features"]))
        return self

    # ----- serve (online lookups, same code path for train & predict) ------ #
    def get_user_features(self, user_ids) -> pd.DataFrame:
        return self.user_features[self.user_features["user_id"].isin(user_ids)]

    def get_item_features(self, item_ids) -> pd.DataFrame:
        return self.item_features[self.item_features["item_id"].isin(item_ids)]


def run(prep: dict | None = None, cfg: dict | None = None) -> FeatureStore:
    cfg = cfg or load_config()
    if prep is None:
        from src.preparation import prepare
        prep = prepare.run(cfg=cfg)
    store = FeatureStore(cfg).build(prep)
    store.save()
    logger.info(
        "Feature store built: %d users x %d user-features | %d items x %d item-features",
        len(store.user_features), store.user_features.shape[1],
        len(store.item_features), store.item_features.shape[1],
    )
    return store


if __name__ == "__main__":
    run()
