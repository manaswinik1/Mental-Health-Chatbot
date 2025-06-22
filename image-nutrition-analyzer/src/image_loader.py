"""Utilities for loading and preprocessing food images."""

from typing import Optional

import cv2
import numpy as np


def load_image(path: str, img_size: int = 640) -> np.ndarray:
    """Load an image from ``path`` and resize for model input.

    Parameters
    ----------
    path : str
        Path to the image file.
    img_size : int, optional
        Target size for the longest image dimension. Defaults to ``640``.

    Returns
    -------
    np.ndarray
        The loaded and resized image in RGB format.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Convert from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    scale = img_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized

