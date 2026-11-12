"""Recommender models.

Two models, a common interface:

  * PopularityRecommender      - non-personalised baseline. Recommends the
                                 globally most popular items (by interaction
                                 count), minus items the user already saw.

  * MatrixFactorizationRecommender - personalised. FunkSVD-style latent factor
                                 model with global/user/item biases, trained by
                                 SGD on observed ratings. Predicts explicit
                                 ratings (-> RMSE/MAE) and ranks unseen items
                                 by predicted rating (-> ranking metrics).

Both expose:
    fit(train_df)
    recommend(user_id, k, seen) -> list[item_id]
    predict(user_id, item_id)   -> float   (MF only returns meaningful values)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class BaseRecommender:
    def fit(self, train: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def recommend(self, user_id: int, k: int, seen: set | None = None) -> list[int]:
        raise NotImplementedError

    def predict(self, user_id: int, item_id: int) -> float:
        raise NotImplementedError


@dataclass
class PopularityRecommender(BaseRecommender):
    """Global popularity baseline."""
    name: str = "popularity_baseline"
    ranked_items_: list[int] = field(default_factory=list)
    item_mean_: dict = field(default_factory=dict)
    global_mean_: float = 0.0

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        counts = train.groupby("item_id").size().sort_values(ascending=False)
        self.ranked_items_ = counts.index.tolist()
        self.item_mean_ = train.groupby("item_id")["rating"].mean().to_dict()
        self.global_mean_ = float(train["rating"].mean())
        return self

    def recommend(self, user_id: int, k: int, seen: set | None = None) -> list[int]:
        seen = seen or set()
        recs = [i for i in self.ranked_items_ if i not in seen]
        return recs[:k]

    def predict(self, user_id: int, item_id: int) -> float:
        # Non-personalised: fall back to the item's mean rating.
        return float(self.item_mean_.get(item_id, self.global_mean_))


@dataclass
class MatrixFactorizationRecommender(BaseRecommender):
    """FunkSVD with biases, trained by SGD.

    r_hat(u,i) = mu + b_u + b_i + p_u . q_i
    """
    n_factors: int = 32
    n_epochs: int = 30
    lr: float = 0.01
    reg: float = 0.05
    seed: int = 42
    name: str = "matrix_factorization"

    def __post_init__(self):
        self.user_index_: dict = {}
        self.item_index_: dict = {}
        self.items_: np.ndarray = np.array([])
        self.mu_: float = 0.0
        self.bu_: np.ndarray = np.array([])
        self.bi_: np.ndarray = np.array([])
        self.P_: np.ndarray = np.array([])
        self.Q_: np.ndarray = np.array([])
        self.train_loss_: list[float] = []

    def fit(self, train: pd.DataFrame) -> "MatrixFactorizationRecommender":
        rng = np.random.default_rng(self.seed)
        users = train["user_id"].unique()
        items = train["item_id"].unique()
        self.user_index_ = {u: idx for idx, u in enumerate(users)}
        self.item_index_ = {i: idx for idx, i in enumerate(items)}
        self.items_ = items

        n_u, n_i = len(users), len(items)
        self.mu_ = float(train["rating"].mean())
        self.bu_ = np.zeros(n_u)
        self.bi_ = np.zeros(n_i)
        self.P_ = rng.normal(0, 0.1, size=(n_u, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, size=(n_i, self.n_factors))

        u_arr = train["user_id"].map(self.user_index_).to_numpy()
        i_arr = train["item_id"].map(self.item_index_).to_numpy()
        r_arr = train["rating"].to_numpy(dtype=float)
        n = len(r_arr)

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            sq_err = 0.0
            for idx in order:
                u, i, r = u_arr[idx], i_arr[idx], r_arr[idx]
                pred = self.mu_ + self.bu_[u] + self.bi_[i] + self.P_[u] @ self.Q_[i]
                err = r - pred
                sq_err += err * err
                # SGD updates.
                self.bu_[u] += self.lr * (err - self.reg * self.bu_[u])
                self.bi_[i] += self.lr * (err - self.reg * self.bi_[i])
                pu = self.P_[u].copy()
                self.P_[u] += self.lr * (err * self.Q_[i] - self.reg * self.P_[u])
                self.Q_[i] += self.lr * (err * pu - self.reg * self.Q_[i])
            self.train_loss_.append(float(np.sqrt(sq_err / n)))
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        u = self.user_index_.get(user_id)
        i = self.item_index_.get(item_id)
        if u is None and i is None:
            return self.mu_
        if u is None:
            return float(self.mu_ + self.bi_[i])
        if i is None:
            return float(self.mu_ + self.bu_[u])
        return float(self.mu_ + self.bu_[u] + self.bi_[i] + self.P_[u] @ self.Q_[i])

    def _scores_for_user(self, user_id: int) -> np.ndarray:
        u = self.user_index_.get(user_id)
        if u is None:
            return self.mu_ + self.bi_  # cold user -> item-bias ranking
        return self.mu_ + self.bu_[u] + self.bi_ + self.P_[u] @ self.Q_.T

    def recommend(self, user_id: int, k: int, seen: set | None = None) -> list[int]:
        seen = seen or set()
        scores = self._scores_for_user(user_id)
        order = np.argsort(-scores)
        recs: list[int] = []
        for idx in order:
            item = int(self.items_[idx])
            if item in seen:
                continue
            recs.append(item)
            if len(recs) == k:
                break
        return recs
