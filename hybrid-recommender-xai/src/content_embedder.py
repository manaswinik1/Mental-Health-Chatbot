from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_content_embeddings(
    metadata_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    existing_model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """Embed item descriptions using a sentence transformer model.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing at least a ``description`` column.
    model_name : str, optional
        Name of the pretrained model to load when ``existing_model`` is not
        provided.
    existing_model : SentenceTransformer, optional
        Preloaded model instance to reuse across calls.

    Returns
    -------
    np.ndarray
        Array of text embeddings aligned with ``metadata_df`` index order.
    """
    model = existing_model or SentenceTransformer(model_name)
    descriptions = metadata_df["description"].fillna("").tolist()
    embeddings = model.encode(descriptions, show_progress_bar=False)
    return embeddings
