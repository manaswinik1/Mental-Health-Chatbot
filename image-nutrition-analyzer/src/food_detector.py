"""Food detection utilities using a YOLOv5 model."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Path to local YOLOv5 weights
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov5s.pt"


class FoodDetector:
    """Wrapper around a YOLOv5 model for food detection."""

    def __init__(self, model_path: Path = MODEL_PATH, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load YOLOv5 model from local weights
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(model_path),
            force_reload=False,
        ).to(self.device)

    def __call__(self, image: np.ndarray) -> List[Dict]:
        """Run detection on a given image.

        Parameters
        ----------
        image : np.ndarray
            Image array in RGB format.

        Returns
        -------
        List[Dict]
            A list of detections with bounding boxes, labels, and confidences.
        """
        results = self.model(image)
        detections = []
        df = results.pandas().xyxy[0]
        for _, row in df.iterrows():
            detections.append(
                {
                    "bbox": [
                        float(row["xmin"]),
                        float(row["ymin"]),
                        float(row["xmax"]),
                        float(row["ymax"]),
                    ],
                    "confidence": float(row["confidence"]),
                    "label": row["name"],
                }
            )
        return detections


# Singleton instance used by the helper function
_detector: FoodDetector | None = None


def detect_food(image: np.ndarray) -> List[Dict]:
    """Detect food items in ``image`` using a YOLOv5 model."""
    global _detector
    if _detector is None:
        _detector = FoodDetector()
    return _detector(image)

