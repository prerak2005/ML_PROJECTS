# Movie Recommendation System

A beginner-level recommender on the MovieLens dataset. It predicts how a user would rate
unseen movies and produces a personalised top-N list, comparing collaborative filtering,
a content-based embedding model, and a hybrid blend against a popularity baseline.

## Problem Statement

Given a sparse matrix of user × movie ratings (0.5 to 5.0), recommend movies each user is
likely to enjoy. There is **no external label** — this is self-supervised: we hold out
known ratings and measure how well each model recovers them, both as a predicted rating
(**RMSE**) and as a ranked recommendation list (**Precision@10 / Recall@10 / NDCG@10**).

## Dataset

- **Source**: [MovieLens (small). GroupLens](https://grouplens.org/datasets/movielens/) (`ml-latest-small`)
- **Ratings**: 100,836 · **Users**: 610 · **Movies**: 9,724
- The full user-item matrix is **~98.3% empty**; sparsity is the core challenge.

| File | Contents |
|---|---|
| `data/ratings.csv` | userId, movieId, rating, timestamp |
| `data/movies.csv` | movieId, title, genres |
| `data/train.csv` / `test.csv` | Hold-out split from notebook 02 |
| `data/item_embeddings_minilm.npz` | Cached MiniLM embeddings (notebook 04) |

For a tractable matrix the modelling restricts to active users and the **2,000 most-rated
movies** (a popularity cut that mitigates extreme sparsity).

## Project Structure

```
movie_recommendation/
├── 01_eda.ipynb              # Ratings, long-tail activity, sparsity, top movies
├── 02_data_cleaning.ipynb    # Filter users/movies, matrix, train/test split
├── 03_model_building.ipynb   # Popularity / Item-CF / SVD (baseline results)
├── 04_advanced_models.ipynb  # Rigorous eval, content embeddings, hybrid, analysis
├── utils.py                  # Loaders, models, evaluation & analysis helpers
├── requirements.txt
├── README.md
└── data/
    ├── ratings.csv
    ├── movies.csv
    ├── train.csv
    ├── test.csv
    └── item_embeddings_minilm.npz
```

Run order: `01` → `02` → `03` → `04`. Notebooks 01–03 are unchanged baselines;
new work lives in `04_advanced_models.ipynb` and `utils.py`.

## Models

- **Popularity baseline** — each movie’s global mean rating (no personalisation).
- **Item-based CF** — cosine similarity on mean-centred ratings; weighted-average predict.
- **Matrix factorization (SVD)** — TruncatedSVD with 20 latent factors on the
  mean-centred, zero-filled matrix (`random_state=42`).
- **Content-Emb** — `all-MiniLM-L6-v2` embeddings of `title + genres`; user profile =
  rating-weighted mean of train likes (≥ 4.0); rank by cosine similarity. Embeddings
  cached to disk.
- **Hybrid(α=0.6)** — per-user min-max blend
  `α · SVD + (1 − α) · Content`. **α = 0.6** chosen by maximising NDCG@10 on a
  **train-only** validation split (never the test set).

## Evaluation protocol

Documented in `04_advanced_models.ipynb` (Phase 1). Used for all ranking numbers below.

| Rule | Setting |
|---|---|
| Split | Random rating hold-out from notebook 02 (`seed=42`); 62,711 train / 15,594 test |
| Relevant item | Test rating **≥ 4.0** |
| Candidates | Catalog movies the user did **not** rate in train |
| Users evaluated | Users with ≥ 1 relevant test item (**591**; 0 skipped) |
| Ranking metrics | Precision@10, Recall@10, **NDCG@10** (binary gains) |
| Leakage check | No `(user, item)` overlap between train and test; test cells absent from the train matrix used for fitting — **PASS** |
| Hybrid α | Selected on train→val only (`val_size=0.2`, `seed=42`); evaluated once on test |

## Results

Baseline RMSE / P@10 / R@10 match `03_model_building.ipynb`. NDCG@10, Content-Emb, and
Hybrid figures are from executing `04_advanced_models.ipynb` — not assumed.

Modelling matrix: **610 users × 2,000 movies** (94.9% sparse).

| Model | RMSE ↓ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ |
|---|---|---|---|---|
| Popularity | 0.9381 | 0.0146 | 0.0119 | 0.0148 |
| Item-based CF | **0.8298** | 0.0262 | 0.0107 | 0.0250 |
| SVD (20 factors) | 0.8918 | **0.1293** | 0.1200 | 0.1755 |
| Content-Emb | 1.1465 | 0.0394 | 0.0427 | 0.0487 |
| **Hybrid(α=0.6)** | 0.9446 | **0.1293** | **0.1293** | **0.1843** |

### Activity / cold-start (Phase 4)

Train-activity tertiles among the 591 evaluated users: **light ≤ 33** (n=203),
**medium ≤ 91** (n=192), **heavy > 91** (n=196). Cold-start = **≤ 20** train ratings
(84 evaluable users).

| Cohort | SVD NDCG@10 | Content NDCG@10 | Hybrid NDCG@10 |
|---|---|---|---|
| Light | 0.1044 | 0.0408 | **0.1230** |
| Medium | 0.1431 | 0.0403 | **0.1629** |
| Heavy | **0.2808** | 0.0651 | 0.2686 |
| Cold (≤20) | 0.0749 | 0.0394 | **0.0971** |

## Key Findings

Only claims supported by the tables above:

- **No single model wins every metric.** Item-CF has the best **RMSE (0.8298)**; ranking
  is led by **SVD / Hybrid**.
- **Content-Emb alone does not beat SVD** on P@10, R@10, or NDCG@10 (0.039 / 0.043 / 0.049
  vs 0.129 / 0.120 / 0.176). Title+genres text is a weak signal on this catalog, but it
  still beats Popularity and Item-CF on ranking.
- **Hybrid(α=0.6) improves ranking over SVD** on R@10 (0.129 vs 0.120) and NDCG@10
  (0.184 vs 0.176), and ties P@10 (0.129). α was not retuned on test.
- **Hybrid helps sparse users more than heavy ones.** On light and cold cohorts Hybrid
  beats SVD on NDCG@10; on heavy users **SVD still wins** (0.281 vs 0.269). Content alone
  never beats SVD in these cohorts.
- **Popularity is a weak floor** on both rating and ranking metrics.
- **Pick the metric that matches the product:** RMSE if you show predicted stars; ranking
  metrics if you show a top-N list. Hybrid RMSE is secondary (blended raw predicts).

## Limitations

- Small MovieLens slice after a 2,000-movie popularity cut — not production scale.
- Offline hold-out metrics only; no online A/B or engagement labels.
- Popularity cut biases toward blockbusters and softens true long-tail cold-start.
- Content uses title + genres only (no plots/cast); embeddings are cached MiniLM.
- No true zero-history user cold-start (min train ratings after filters: 6).
- Hybrid α fixed once on train-val; not re-tuned per cohort.

## Tech Stack

- pandas, numpy, matplotlib, seaborn
- scikit-learn
- sentence-transformers (`all-MiniLM-L6-v2`)

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```

First content-embedding run downloads MiniLM and writes
`data/item_embeddings_minilm.npz`; later runs load the cache.

## What I would do next

Not implemented — directions for a stronger portfolio / production path:

- Serve recommendations behind a small FastAPI (or similar) with cached user/item vectors.
- Add an **LLM re-ranking** step over a short candidate list (e.g. top-50 from Hybrid) using
  richer metadata or user taste text — retrieval + ranking, not end-to-end generation.
- Stronger content: plot summaries / tags from an external catalog; evaluate whether
  Content alone then closes more of the gap vs SVD.
- Explicit feedback loop / online evaluation; fairness and popularity-bias diagnostics
  beyond the offline tables above.
- Better cold-start: content-only path for users with zero history; item cold-start for
  brand-new movies.
