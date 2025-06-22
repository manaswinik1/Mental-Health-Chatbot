"""Data loading utilities for the Mental Health Chatbot project."""

from typing import List, Dict
import pandas as pd


def load_data(path: str = "data/raw/empathetic_dialogues.csv") -> List[Dict[str, str]]:
    """Load and clean the empathetic dialogue dataset.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file containing the dialogues, by default
        "data/raw/empathetic_dialogues.csv".

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries with keys ``prompt`` and ``utterance``.
    """
    df = pd.read_csv(path)

    # Keep only necessary columns
    df = df[["prompt", "utterance"]]

    # Drop rows with missing or empty values
    df.dropna(subset=["prompt", "utterance"], inplace=True)
    df = df[df["prompt"].str.strip() != ""]
    df = df[df["utterance"].str.strip() != ""]

    return df.to_dict("records")


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} prompt-response pairs.")
