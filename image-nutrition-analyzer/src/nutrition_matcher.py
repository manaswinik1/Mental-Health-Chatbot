"""Match detected food items with nutritional information."""

from pathlib import Path
from typing import List

import pandas as pd

# Path to nutrition lookup table
LOOKUP_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "nutrition_lookup.csv"


class NutritionMatcher:
    """Utility class to match detections to nutritional data."""

    def __init__(self, csv_path: Path = LOOKUP_PATH) -> None:
        self.table = pd.read_csv(csv_path)

    def match(self, labels: List[str]) -> pd.DataFrame:
        """Return nutrition info for the provided labels."""
        labels_lower = [lbl.lower() for lbl in labels]
        matched = self.table[self.table["item_name"].str.lower().isin(labels_lower)]
        # Aggregate duplicates
        summary = (
            matched.groupby("item_name", as_index=False)[
                ["calories", "protein_g", "fat_g", "carbs_g"]
            ].sum()
        )
        return summary


_matcher: NutritionMatcher | None = None


def match_nutrition(detected_items: List[str]) -> pd.DataFrame:
    """Helper wrapper for :class:`NutritionMatcher`."""
    global _matcher
    if _matcher is None:
        _matcher = NutritionMatcher()
    return _matcher.match(detected_items)

