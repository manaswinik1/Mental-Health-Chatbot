"""Data loading utilities for stock prices and news headlines."""

from pathlib import Path
from typing import Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
STOCK_PATH = ROOT_DIR / "data" / "raw" / "stock_prices.csv"
NEWS_PATH = ROOT_DIR / "data" / "raw" / "news_headlines.csv"


def load_stock_data(path: Path = STOCK_PATH) -> pd.DataFrame:
    """Load stock prices CSV with parsed dates."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_news_data(path: Path = NEWS_PATH) -> pd.DataFrame:
    """Load news headlines CSV with parsed dates."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    stock_df = load_stock_data()
    news_df = load_news_data()
    print(f"Loaded {len(stock_df)} rows of stock data")
    print(f"Loaded {len(news_df)} rows of news data")
