"""Format nutrition information for presentation."""

from typing import Tuple
import io

import matplotlib.pyplot as plt
import pandas as pd


def summarize_nutrition(nutrition_df: pd.DataFrame) -> Tuple[str, bytes]:
    """Create a summary text and pie chart of macronutrients.

    Parameters
    ----------
    nutrition_df : pd.DataFrame
        DataFrame returned by ``match_nutrition``.

    Returns
    -------
    Tuple[str, bytes]
        Human readable text summary and PNG bytes of a pie chart.
    """
    totals = nutrition_df[["calories", "protein_g", "fat_g", "carbs_g"]].sum()
    text = (
        f"Total Calories: {totals['calories']:.1f} kcal\n"
        f"Protein: {totals['protein_g']:.1f} g\n"
        f"Fat: {totals['fat_g']:.1f} g\n"
        f"Carbs: {totals['carbs_g']:.1f} g"
    )

    fig, ax = plt.subplots()
    labels = ["Protein", "Fat", "Carbs"]
    values = [totals["protein_g"], totals["fat_g"], totals["carbs_g"]]
    ax.pie(values, labels=labels, autopct="%1.1f%%")
    ax.set_title("Macronutrient Distribution")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return text, buf.getvalue()

