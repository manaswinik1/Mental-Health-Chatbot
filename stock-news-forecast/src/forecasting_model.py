"""Model training and evaluation for stock forecasting."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rf_model.pkl"


def train_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.Series, pd.Series]:
    """Train a RandomForest model and return predictions."""
    features = df.drop(columns=["date", "close"])
    target = df["close"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    return model, X_test, y_test, y_pred
