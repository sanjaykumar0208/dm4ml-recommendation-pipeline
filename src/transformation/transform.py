"""Stage 4 - Data Transformation (derived datasets).

The transformation logic lives alongside preparation in
``src/preparation/prepare.py`` because the two stages share one clean pass over
the data (the assignment groups them as "Data Preparation and Transformation").
This module exposes the *transformation-specific* builders on their own so they
can be imported, reused and unit-tested independently of the preparation stage:

    * build_user_activity_summary  - per-user counts / averages / recency
    * build_item_popularity        - per-item counts / averages / rank
    * build_user_item_matrix       - users x items rating matrix (from train)

Example
-------
>>> from src.transformation import transform
>>> pop = transform.build_item_popularity(train_df)
"""
from __future__ import annotations

import pandas as pd

from src.preparation.prepare import (
    _item_popularity,
    _user_activity_summary,
    _user_item_matrix,
)


def build_user_activity_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    """Per-user activity summary (count, avg rating, distinct items, recency)."""
    return _user_activity_summary(interactions)


def build_item_popularity(interactions: pd.DataFrame) -> pd.DataFrame:
    """Per-item popularity table (count, avg rating, distinct users, rank)."""
    return _item_popularity(interactions)


def build_user_item_matrix(train: pd.DataFrame) -> pd.DataFrame:
    """User x item rating matrix built from the training interactions."""
    return _user_item_matrix(train)
