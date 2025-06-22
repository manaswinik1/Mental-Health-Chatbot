from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import load_data
from .content_embedder import generate_content_embeddings


def _compute_cf_scores(user_id: int, ratings_df: pd.DataFrame) -> pd.Series:
    """Internal helper to compute collaborative filtering scores."""
    user_item = ratings_df.pivot_table(
        index="user_id", columns="item_id", values="rating"
    )
    rating_matrix = user_item.fillna(0)

    similarity = cosine_similarity(rating_matrix.T)
    sim_df = pd.DataFrame(
        similarity, index=user_item.columns, columns=user_item.columns
    )

    if user_id not in rating_matrix.index:
        return pd.Series(dtype=float)

    user_vector = rating_matrix.loc[user_id]
    scores = sim_df.dot(user_vector).div(sim_df.sum(axis=1))
    rated_items = user_vector[user_vector > 0].index.tolist()
    scores = scores.drop(rated_items, errors="ignore")
    return scores


def _compute_content_scores(
    user_id: int, ratings_df: pd.DataFrame, metadata_df: pd.DataFrame, embeddings: np.ndarray
) -> pd.Series:
    """Internal helper to compute content similarity scores."""
    liked = ratings_df[(ratings_df["user_id"] == user_id) & (ratings_df["rating"] >= 4)]
    if liked.empty:
        return pd.Series(0, index=metadata_df["item_id"], dtype=float)

    indices = metadata_df.index
    liked_embeddings = embeddings[[indices.get_loc(iid) for iid in liked["item_id"]]]
    user_profile = liked_embeddings.mean(axis=0, keepdims=True)
    sims = cosine_similarity(user_profile, embeddings)[0]
    scores = pd.Series(sims, index=metadata_df["item_id"])
    rated_items = ratings_df[ratings_df["user_id"] == user_id]["item_id"].tolist()
    return scores.drop(rated_items, errors="ignore")


def hybrid_recommend(
    user_id: int,
    top_n: int = 10,
    weights: Tuple[float, float] = (0.5, 0.5),
) -> List[int]:
    """Return top-N recommendations combining CF and content similarity."""
    ratings, metadata = load_data()
    embeddings = generate_content_embeddings(metadata)

    cf_scores = _compute_cf_scores(user_id, ratings)
    content_scores = _compute_content_scores(user_id, ratings, metadata, embeddings)

    all_scores = (
        cf_scores.mul(weights[0], fill_value=0)
        .add(content_scores.mul(weights[1], fill_value=0))
    )
    if all_scores.empty:
        return []
    return all_scores.sort_values(ascending=False).head(top_n).index.tolist()
