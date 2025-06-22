from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_cf(
    user_id: int, ratings_df: pd.DataFrame, top_n: int = 10
) -> List[int]:
    """Generate item recommendations using item-based collaborative filtering.

    Parameters
    ----------
    user_id : int
        ID of the user to generate recommendations for.
    ratings_df : pd.DataFrame
        DataFrame containing at least ``user_id``, ``item_id`` and ``rating`` columns.
    top_n : int, optional
        Number of recommendations to return, by default 10.

    Returns
    -------
    List[int]
        List of recommended item IDs sorted by predicted relevance.
    """
    if user_id not in ratings_df["user_id"].unique():
        return []

    user_item = ratings_df.pivot_table(
        index="user_id", columns="item_id", values="rating"
    )
    rating_matrix = user_item.fillna(0)

    # Compute item-item similarity matrix
    similarity = cosine_similarity(rating_matrix.T)
    sim_df = pd.DataFrame(
        similarity, index=user_item.columns, columns=user_item.columns
    )

    user_vector = rating_matrix.loc[user_id]
    scores = sim_df.dot(user_vector).div(sim_df.sum(axis=1))

    rated_items = user_vector[user_vector > 0].index.tolist()
    scores = scores.drop(rated_items, errors="ignore")

    return scores.sort_values(ascending=False).head(top_n).index.tolist()
