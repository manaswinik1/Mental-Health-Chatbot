import pandas as pd
from typing import Tuple


def load_data(
    ratings_path: str = "data/raw/movielens_ratings.csv",
    metadata_path: str = "data/raw/movielens_metadata.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess ratings and metadata.

    Parameters
    ----------
    ratings_path : str, optional
        Path to the ratings CSV file.
    metadata_path : str, optional
        Path to the metadata CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the cleaned ratings dataframe and metadata dataframe
        with textual information.
    """
    # Load CSV files
    ratings = pd.read_csv(ratings_path)
    metadata = pd.read_csv(metadata_path)

    # Basic cleaning
    ratings = ratings.dropna(subset=["user_id", "item_id", "rating"])
    metadata = metadata.dropna(subset=["item_id", "title", "description"])

    # Convert dtypes
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    metadata["item_id"] = metadata["item_id"].astype(int)

    return ratings, metadata
