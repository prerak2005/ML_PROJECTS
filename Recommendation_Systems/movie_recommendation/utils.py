"""
Movie Recommendation System — Utility Functions
Collaborative filtering on the MovieLens (small) explicit-rating dataset.
Two models are compared against a popularity baseline:
  - Item-based CF   : cosine similarity between items on mean-centred ratings
  - Matrix factorization : TruncatedSVD latent factors
Evaluation: RMSE (rating prediction) + Precision@k / Recall@k (top-k ranking).

This is an unsupervised / self-supervised setup — there is no external label;
we hold out known ratings and measure how well each model recovers them.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# Column mapping for this dataset (other explicit projects reuse this file by
# editing only these three names + the loaders below).
UCOL, ICOL, RCOL = "userId", "movieId", "rating"


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_ratings(path="data/ratings.csv"):
    """Load ratings as a tidy frame with columns user / item / rating."""
    df = pd.read_csv(path)
    return df.rename(columns={UCOL: "user", ICOL: "item", RCOL: "rating"})[["user", "item", "rating"]]


def load_items(path="data/movies.csv", id_col="movieId", name_col="title"):
    """Load an item-id → title lookup Series."""
    items = pd.read_csv(path)
    return items.set_index(id_col)[name_col]


def load_movie_metadata(path: str = "data/movies.csv") -> pd.DataFrame:
    """Load movieId / title / genres as columns item, title, genres."""
    meta = pd.read_csv(path)
    return meta.rename(columns={"movieId": "item"})[["item", "title", "genres"]]


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def filter_popular(df, max_users=2000, max_items=2000, min_user=5, min_item=5):
    """
    Keep the most active users and most-rated items (for a tractable, dense-enough
    matrix), after dropping users/items with too few ratings. Returns filtered df.
    """
    vc_i = df["item"].value_counts()
    df = df[df["item"].isin(vc_i[vc_i >= min_item].index)]
    vc_u = df["user"].value_counts()
    df = df[df["user"].isin(vc_u[vc_u >= min_user].index)]
    top_items = df["item"].value_counts().head(max_items).index
    top_users = df["user"].value_counts().head(max_users).index
    return df[df["item"].isin(top_items) & df["user"].isin(top_users)]


def build_user_item(df):
    """Pivot a tidy ratings frame to a (users × items) matrix; unrated = NaN."""
    return df.pivot_table(index="user", columns="item", values="rating")


def train_test_split_ratings(df, test_size=0.2, seed=42):
    """
    Random hold-out of individual ratings. Rows whose user or item is absent from
    the training side are moved back to train so every test pair is predictable.
    """
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(df)) < test_size
    train, test = df[~mask].copy(), df[mask].copy()
    train_users, train_items = set(train["user"]), set(train["item"])
    movable = ~test["user"].isin(train_users) | ~test["item"].isin(train_items)
    train = pd.concat([train, test[movable]])
    test = test[~movable]
    return train, test


def sparsity(matrix):
    """Fraction of empty cells in the user-item matrix."""
    return float(np.isnan(matrix.values).mean())


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class PopularityBaseline:
    """Predict each item's global mean rating (ignores the user)."""
    def fit(self, train_matrix):
        self.item_means = train_matrix.mean(axis=0)
        self.global_mean = float(np.nanmean(train_matrix.values))
        return self

    def predict(self, user, item):
        return self.item_means.get(item, self.global_mean)

    def score_all(self, user):
        return self.item_means.fillna(self.global_mean)


class ItemBasedCF:
    """Item-item cosine similarity on mean-centred ratings; weighted-average predict."""
    def fit(self, train_matrix):
        self.matrix = train_matrix
        self.user_means = train_matrix.mean(axis=1)
        centred = train_matrix.sub(self.user_means, axis=0).fillna(0.0)
        sim = cosine_similarity(centred.T)
        self.sim = pd.DataFrame(sim, index=train_matrix.columns, columns=train_matrix.columns)
        self.global_mean = float(np.nanmean(train_matrix.values))
        return self

    def predict(self, user, item, k=30):
        if user not in self.matrix.index or item not in self.sim.columns:
            return self.global_mean
        rated = self.matrix.loc[user].dropna()
        if rated.empty:
            return self.user_means.get(user, self.global_mean)
        sims = self.sim.loc[item, rated.index]
        top = sims.sort_values(ascending=False).head(k)
        top = top[top > 0]
        if top.sum() == 0:
            return self.user_means.get(user, self.global_mean)
        centred = rated[top.index] - self.user_means[user]
        return float(self.user_means[user] + (top * centred).sum() / top.sum())

    def score_all(self, user):
        """Vectorised predicted rating for every item (positive similarities only)."""
        items = self.matrix.columns
        if user not in self.matrix.index:
            return pd.Series(self.global_mean, index=items)
        rated = self.matrix.loc[user].dropna()
        um = self.user_means.get(user, self.global_mean)
        if rated.empty:
            return pd.Series(um, index=items)
        centred = (rated - um).values
        simpos = np.clip(self.sim.loc[:, rated.index].values, 0, None)
        num = simpos @ centred
        denom = simpos.sum(axis=1)
        pred = np.where(denom > 0, um + num / np.where(denom == 0, 1, denom), um)
        return pd.Series(pred, index=items)


class SVDRecommender:
    """TruncatedSVD matrix factorization on the mean-centred, zero-filled matrix."""
    def fit(self, train_matrix, n_components=20, seed=42):
        self.user_means = train_matrix.mean(axis=1)
        centred = train_matrix.sub(self.user_means, axis=0).fillna(0.0)
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.U = self.svd.fit_transform(centred)
        self.V = self.svd.components_
        self.recon = pd.DataFrame(self.U @ self.V, index=train_matrix.index,
                                  columns=train_matrix.columns).add(self.user_means, axis=0)
        self.global_mean = float(np.nanmean(train_matrix.values))
        return self

    def predict(self, user, item):
        if user in self.recon.index and item in self.recon.columns:
            return float(self.recon.loc[user, item])
        return self.global_mean

    def score_all(self, user):
        if user in self.recon.index:
            return self.recon.loc[user]
        return pd.Series(self.global_mean, index=self.recon.columns)


# ─────────────────────────────────────────────
# Content-based embeddings
# ─────────────────────────────────────────────

def build_item_texts(meta: pd.DataFrame, item_ids) -> list[str]:
    """Build title+genres text for each catalog item_id (same order as item_ids)."""
    lookup = meta.set_index("item")
    texts: list[str] = []
    for item_id in item_ids:
        row = lookup.loc[item_id]
        genres = str(row["genres"]).replace("|", ", ")
        texts.append(f"{row['title']}. Genres: {genres}")
    return texts


def _l2_normalize(mat: np.ndarray, axis: int = 1) -> np.ndarray:
    """L2-normalise rows (or columns); zero rows stay zero."""
    norms = np.linalg.norm(mat, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def load_or_compute_item_embeddings(
    item_ids,
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_path: str = "data/item_embeddings_minilm.npz",
    seed: int = 42,
) -> np.ndarray:
    """
    Return (n_items, dim) embeddings aligned with item_ids.

    Loads from cache_path when item_ids match; otherwise encodes with
    SentenceTransformer (first run downloads the model) and saves a float32 npz.
    """
    from pathlib import Path

    ids_arr = np.asarray(list(item_ids))
    path = Path(cache_path)
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        cached_ids = cached["item_ids"]
        if len(cached_ids) == len(ids_arr) and np.array_equal(cached_ids, ids_arr):
            print(f"embeddings cache HIT: {path}  shape={cached['embeddings'].shape}")
            return cached["embeddings"].astype(np.float32)

    print(f"embeddings cache MISS — encoding {len(texts)} texts with {model_name}")
    np.random.seed(seed)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, item_ids=ids_arr, embeddings=embeddings)
    print(f"embeddings cache SAVED: {path}  shape={embeddings.shape}")
    return embeddings


class ContentEmbeddingRecommender:
    """
    Content-based CF: user profile = rating-weighted mean of liked-item embeddings
    (train rating >= rel_threshold); rank by cosine similarity to catalog items.
    """

    def fit(
        self,
        train_matrix: pd.DataFrame,
        item_embeddings: np.ndarray,
        rel_threshold: float = 4.0,
    ):
        if item_embeddings.shape[0] != train_matrix.shape[1]:
            raise ValueError("item_embeddings rows must match train_matrix columns")
        self.items = train_matrix.columns
        self.item_index = {item: i for i, item in enumerate(self.items)}
        self.emb = _l2_normalize(item_embeddings.astype(np.float32), axis=1)
        self.catalog_mean = _l2_normalize(self.emb.mean(axis=0, keepdims=True), axis=1)[0]
        self.user_means = train_matrix.mean(axis=1)
        self.global_mean = float(np.nanmean(train_matrix.values))
        self.profiles: dict = {}

        for user in train_matrix.index:
            rated = train_matrix.loc[user].dropna()
            liked = rated[rated >= rel_threshold]
            if len(liked) > 0:
                idxs = [self.item_index[i] for i in liked.index]
                weights = liked.values.astype(np.float32)
                profile = (self.emb[idxs] * weights[:, None]).sum(axis=0) / weights.sum()
            elif len(rated) > 0:
                idxs = [self.item_index[i] for i in rated.index]
                profile = self.emb[idxs].mean(axis=0)
            else:
                profile = self.catalog_mean.copy()
            self.profiles[user] = _l2_normalize(profile.reshape(1, -1), axis=1)[0]
        return self

    def _profile(self, user) -> np.ndarray:
        if user in self.profiles:
            return self.profiles[user]
        return self.catalog_mean

    def score_all(self, user) -> pd.Series:
        """Cosine similarity of the user profile to every catalog item."""
        sims = self.emb @ self._profile(user)
        return pd.Series(sims, index=self.items)

    def predict(self, user, item) -> float:
        """Map cosine into a soft rating: user_mean + cosine, clipped to [0.5, 5.0]."""
        um = float(self.user_means.get(user, self.global_mean))
        if item not in self.item_index:
            return um
        sim = float(self.emb[self.item_index[item]] @ self._profile(user))
        return float(np.clip(um + sim, 0.5, 5.0))


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def rmse_on_test(model, test_df, **kw):
    """Root mean squared error of predicted vs actual on held-out rating pairs."""
    preds = np.array([model.predict(u, i, **kw) if kw else model.predict(u, i)
                      for u, i in zip(test_df["user"], test_df["item"])])
    actual = test_df["rating"].values
    return float(np.sqrt(np.mean((preds - actual) ** 2)))


def precision_recall_at_k(model, train_matrix, test_df, k=10, rel_threshold=4.0):
    """
    Ranking quality via top-k. For each user with held-out 'relevant' items
    (test rating >= rel_threshold), score every unseen item (vectorised), take the
    top-k, and measure precision@k and recall@k; returns the mean over users.
    """
    rel_by_user = (test_df[test_df["rating"] >= rel_threshold]
                   .groupby("user")["item"].apply(set).to_dict())
    precs, recs = [], []
    for user, relevant in rel_by_user.items():
        if user not in train_matrix.index:
            continue
        scores = model.score_all(user).copy()
        seen = train_matrix.loc[user].dropna().index
        scores = scores.drop(index=seen, errors="ignore")
        topk = set(scores.sort_values(ascending=False).head(k).index)
        hits = len(topk & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
    return {"precision@k": round(np.mean(precs), 4), "recall@k": round(np.mean(recs), 4),
            "k": k, "n_users_evaluated": len(precs)}


def recommend(model, train_matrix, user, items_lookup=None, n=10):
    """Top-n unseen-item recommendations for a user, with optional title lookup."""
    if user not in train_matrix.index:
        return pd.DataFrame()
    scores = model.score_all(user).copy()
    seen = train_matrix.loc[user].dropna().index
    scores = scores.drop(index=seen, errors="ignore").sort_values(ascending=False).head(n)
    out = pd.DataFrame({"item": scores.index, "score": scores.values.round(3)})
    if items_lookup is not None and not out.empty:
        out.insert(1, "title", out["item"].map(items_lookup))
    return out


def _dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain for a binary relevance list (rank starts at 1)."""
    return float(sum(rel / np.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1)))


def ranking_metrics_at_k(
    model,
    train_matrix: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    rel_threshold: float = 4.0,
) -> dict:
    """
    Ranking metrics for top-k recommendation lists.

    Protocol
    --------
    - Relevant: a test rating is relevant if rating >= rel_threshold (default 4.0).
    - Candidates: every movie in train_matrix columns that the user did NOT rate
      in train. Held-out test items that were never rated in train remain candidates.
    - Users evaluated: users with >= 1 relevant test item AND present in
      train_matrix.index. Users with relevant test items but no train row are
      skipped and counted in n_users_skipped.
    - Metrics (mean over evaluated users): Precision@k, Recall@k, NDCG@k.
      NDCG uses binary gains; IDCG places min(k, n_relevant) hits in the top ranks.
    """
    rel_by_user = (
        test_df[test_df["rating"] >= rel_threshold]
        .groupby("user")["item"]
        .apply(set)
        .to_dict()
    )
    evaluated_users: list = []
    n_users_skipped = 0
    precs, recs, ndcgs = [], [], []

    for user, relevant in sorted(rel_by_user.items()):
        if user not in train_matrix.index:
            n_users_skipped += 1
            continue
        scores = model.score_all(user).copy()
        seen = train_matrix.loc[user].dropna().index
        scores = scores.drop(index=seen, errors="ignore")
        ranked = scores.sort_values(ascending=False).head(k)
        topk_list = list(ranked.index)
        topk_set = set(topk_list)
        hits = len(topk_set & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
        gains = [1 if item in relevant else 0 for item in topk_list]
        dcg = _dcg_at_k(gains, k)
        ideal = _dcg_at_k([1] * min(k, len(relevant)), k)
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        evaluated_users.append(user)

    return {
        "precision@k": float(np.mean(precs)) if precs else 0.0,
        "recall@k": float(np.mean(recs)) if recs else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "k": k,
        "rel_threshold": rel_threshold,
        "n_users_evaluated": len(evaluated_users),
        "n_users_skipped": n_users_skipped,
        "evaluated_users": evaluated_users,
    }


def leakage_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_matrix: pd.DataFrame,
) -> dict:
    """
    Confirm the train/test split does not leak into model fitting.

    Checks
    ------
    1. No (user, item) pair appears in both train_df and test_df.
    2. No test (user, item) cell is already filled in train_matrix
       (item means / SVD see only this matrix; a filled cell would mean the
       test rating entered the fit). Zero-filling missing cells is not leakage.

    Prints PASS/FAIL and returns the counts.
    """
    train_pairs = set(zip(train_df["user"], train_df["item"]))
    test_pairs = set(zip(test_df["user"], test_df["item"]))
    n_overlap_pairs = len(train_pairs & test_pairs)

    n_test_in_matrix = 0
    for user, item in test_pairs:
        if user in train_matrix.index and item in train_matrix.columns:
            if pd.notna(train_matrix.loc[user, item]):
                n_test_in_matrix += 1

    passed = n_overlap_pairs == 0 and n_test_in_matrix == 0
    status = "PASS" if passed else "FAIL"
    print(
        f"leakage_check: {status} | "
        f"overlap_pairs={n_overlap_pairs} | "
        f"test_cells_filled_in_train_matrix={n_test_in_matrix}"
    )
    return {
        "passed": passed,
        "n_overlap_pairs": n_overlap_pairs,
        "n_test_in_matrix": n_test_in_matrix,
    }


# ─────────────────────────────────────────────
# Hybrid SVD + content
# ─────────────────────────────────────────────

def minmax_normalize(scores: pd.Series) -> pd.Series:
    """Map scores to [0, 1] per series; constant series becomes all zeros."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi == lo:
        return pd.Series(0.0, index=scores.index)
    return (scores - lo) / (hi - lo)


class HybridRecommender:
    """
    Blend SVD and content scores: alpha * svd_norm + (1 - alpha) * content_norm.

    Alpha should be chosen on a train-only validation split (never the test set).
    Ranking uses per-user min-max normalised score_all outputs; predict() blends
    raw model predicts for optional RMSE only.
    """

    def __init__(self, svd_model, content_model, alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.svd = svd_model
        self.content = content_model
        self.alpha = float(alpha)

    def score_all(self, user) -> pd.Series:
        svd_n = minmax_normalize(self.svd.score_all(user))
        content_n = minmax_normalize(self.content.score_all(user))
        return self.alpha * svd_n + (1.0 - self.alpha) * content_n

    def predict(self, user, item) -> float:
        """Blend raw predicts (not min-max) for a soft RMSE figure."""
        return float(
            self.alpha * self.svd.predict(user, item)
            + (1.0 - self.alpha) * self.content.predict(user, item)
        )


def select_hybrid_alpha(
    train_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    item_ids=None,
    alphas=None,
    k: int = 10,
    rel_threshold: float = 4.0,
    val_size: float = 0.2,
    seed: int = 42,
    n_components: int = 20,
) -> dict:
    """
    Choose hybrid alpha on a validation split carved from train only.

    Fits SVD + Content on the fit half, sweeps alphas by NDCG@k on val
    (ties → higher P@k, then higher alpha). Never reads the held-out test set.

    item_ids: catalog ids aligned with item_embeddings rows (typically full-train
    M.columns). Required when the fit matrix has a different column order/set.
    """
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 11)

    fit_df, val_df = train_test_split_ratings(train_df, test_size=val_size, seed=seed)
    M_fit = build_user_item(fit_df)

    if item_ids is None:
        if item_embeddings.shape[0] != M_fit.shape[1]:
            raise ValueError(
                "item_embeddings rows must match M_fit columns, or pass item_ids "
                "aligned to the full-catalog embeddings"
            )
        emb_fit = item_embeddings
    else:
        id_to_row = {iid: i for i, iid in enumerate(item_ids)}
        try:
            rows = [id_to_row[iid] for iid in M_fit.columns]
        except KeyError as exc:
            raise ValueError(f"item_ids missing catalog item: {exc}") from exc
        emb_fit = item_embeddings[rows]

    svd_fit = SVDRecommender().fit(M_fit, n_components=n_components, seed=seed)
    content_fit = ContentEmbeddingRecommender().fit(
        M_fit, emb_fit, rel_threshold=rel_threshold
    )

    rows = []
    for alpha in alphas:
        hybrid = HybridRecommender(svd_fit, content_fit, alpha=float(alpha))
        rank = ranking_metrics_at_k(
            hybrid, M_fit, val_df, k=k, rel_threshold=rel_threshold
        )
        rows.append(
            {
                "alpha": float(alpha),
                "P@10": rank["precision@k"],
                "R@10": rank["recall@k"],
                "NDCG@10": rank["ndcg@k"],
                "n_users": rank["n_users_evaluated"],
            }
        )

    val_curve = pd.DataFrame(rows)
    ordered = val_curve.sort_values(
        by=["NDCG@10", "P@10", "alpha"], ascending=[False, False, False]
    )
    best = ordered.iloc[0]
    best_alpha = round(float(best["alpha"]), 1)
    print(
        f"select_hybrid_alpha: best_alpha={best_alpha:.1f} | "
        f"val NDCG@10={best['NDCG@10']:.4f} | "
        f"val P@10={best['P@10']:.4f} | n_val_users={int(best['n_users'])}"
    )
    return {
        "best_alpha": best_alpha,
        "val_curve": val_curve,
        "n_fit": len(fit_df),
        "n_val": len(val_df),
    }


# ─────────────────────────────────────────────
# Phase 4: activity / cold-start analysis
# ─────────────────────────────────────────────

def per_user_ranking_at_k(
    model,
    train_matrix: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    rel_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Per-user ranking metrics under the Phase 1 protocol.

    Returns one row per evaluated user: user, precision@k, recall@k, ndcg@k.
    """
    rel_by_user = (
        test_df[test_df["rating"] >= rel_threshold]
        .groupby("user")["item"]
        .apply(set)
        .to_dict()
    )
    rows = []
    for user, relevant in sorted(rel_by_user.items()):
        if user not in train_matrix.index:
            continue
        scores = model.score_all(user).copy()
        seen = train_matrix.loc[user].dropna().index
        scores = scores.drop(index=seen, errors="ignore")
        ranked = scores.sort_values(ascending=False).head(k)
        topk_list = list(ranked.index)
        hits = len(set(topk_list) & relevant)
        gains = [1 if item in relevant else 0 for item in topk_list]
        dcg = _dcg_at_k(gains, k)
        ideal = _dcg_at_k([1] * min(k, len(relevant)), k)
        rows.append(
            {
                "user": user,
                "precision@k": hits / k,
                "recall@k": hits / len(relevant),
                "ndcg@k": (dcg / ideal) if ideal > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def train_activity_counts(train_df: pd.DataFrame) -> pd.Series:
    """Return user → number of train ratings."""
    return train_df.groupby("user").size().rename("n_train")


def assign_activity_tertiles(counts: pd.Series, users) -> dict:
    """
    Label users as light / medium / heavy by tertiles of their train counts.

    Tertile edges are computed on the provided users only. Returns
    {labels: Series, edges: (t1, t2), counts_per_bin: dict}.
    """
    subset = counts.reindex(list(users)).dropna()
    t1, t2 = np.quantile(subset.values, [1 / 3, 2 / 3])

    def _label(n: float) -> str:
        if n <= t1:
            return "light"
        if n <= t2:
            return "medium"
        return "heavy"

    labels = subset.map(_label)
    labels = labels.astype(
        pd.CategoricalDtype(categories=["light", "medium", "heavy"], ordered=True)
    )
    counts_per_bin = labels.value_counts().reindex(["light", "medium", "heavy"]).fillna(0).astype(int)
    print(
        f"activity tertiles: light ≤ {t1:.0f} | "
        f"medium ≤ {t2:.0f} | heavy > {t2:.0f} | "
        f"n={dict(counts_per_bin)}"
    )
    return {"labels": labels, "edges": (float(t1), float(t2)), "counts_per_bin": dict(counts_per_bin)}
