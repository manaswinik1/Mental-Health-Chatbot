"""Streamlit dashboard for stock price forecasting."""

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_stock_data, load_news_data
from src.sentiment_analyzer import analyze_sentiment
from src.feature_engineering import generate_features
from src.forecasting_model import train_model


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_model(path: Path) -> object:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def main() -> None:
    st.title("Stock Forecasting with News Fusion")

    stock_df = load_stock_data()
    news_df = load_news_data()
    news_df = analyze_sentiment(news_df)
    feature_df = generate_features(stock_df, news_df)

    model_option = st.selectbox("Model", ["RandomForest"])
    if st.button("Train / Reload Model"):
        model, X_test, y_test, y_pred = train_model(feature_df)
        with open(ROOT_DIR / "models" / "rf_model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        model = load_model(ROOT_DIR / "models" / "rf_model.pkl")
        if model:
            features = feature_df.drop(columns=["date", "close"])
            target = feature_df["close"]
            y_pred = model.predict(features)
            y_test = target
        else:
            st.warning("Model not found. Please train first.")
            return

    st.subheader("Forecast vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_df["date"], y=y_test, name="Actual"))
    fig.add_trace(go.Scatter(x=feature_df["date"], y=y_pred, name="Predicted"))
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
