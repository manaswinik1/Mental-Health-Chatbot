"""Sentiment analysis utilities using FinBERT."""

from typing import List

import pandas as pd
from transformers import pipeline


def analyze_sentiment(news_df: pd.DataFrame, model_name: str = "ProsusAI/finbert") -> pd.DataFrame:
    """Add a sentiment_score column to the news DataFrame."""
    classifier = pipeline("sentiment-analysis", model=model_name)

    def score_headline(headline: str) -> float:
        if not isinstance(headline, str) or headline.strip() == "":
            return 0.0
        result = classifier(headline)[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            return score
        if label == "negative":
            return -score
        return 0.0

    news_df = news_df.copy()
    news_df["sentiment_score"] = news_df["headline"].apply(score_headline)
    return news_df
