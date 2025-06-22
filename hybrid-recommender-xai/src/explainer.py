from typing import Dict

import numpy as np
import shap


def explain_recommendation(cf_score: float, content_score: float) -> str:
    """Generate a simple SHAP-based explanation.

    Parameters
    ----------
    cf_score : float
        Score derived from collaborative filtering.
    content_score : float
        Score derived from content similarity.

    Returns
    -------
    str
        Textual explanation describing the contribution of each component.
    """
    model_output = cf_score + content_score
    features = np.array([[cf_score, content_score]])

    # Simple linear model for demonstration
    linear_weights = np.array([1.0, 1.0])
    explainer = shap.LinearExplainer(linear_weights, features)
    shap_values = explainer.shap_values(features)[0]

    explanation = (
        f"CF contribution: {shap_values[0]:.3f}, "
        f"Content contribution: {shap_values[1]:.3f}, "
        f"Overall score: {model_output:.3f}"
    )
    return explanation
