"""Utilities for merging data and creating features for forecasting."""

from typing import Tuple
import pandas as pd


def generate_features(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge stock and sentiment data, create lag features and moving averages."""
    stock_df = stock_df.copy()
    sentiment_daily = sentiment_df.groupby("date")["sentiment_score"].mean().reset_index()

    df = pd.merge(stock_df, sentiment_daily, on="date", how="left")
    df.sort_values("date", inplace=True)
    df["sentiment_score"].fillna(0.0, inplace=True)

    # Lag features of closing price
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)

    # Moving averages
    df["ma_3"] = df["close"].rolling(window=3).mean()
    df["ma_7"] = df["close"].rolling(window=7).mean()

    df.dropna(inplace=True)
    return df
