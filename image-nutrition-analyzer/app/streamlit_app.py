"""Streamlit interface for the Image-Based Nutritional Analyzer."""

from pathlib import Path
from typing import List

import sys

import cv2
import numpy as np
import streamlit as st

# Add src directory to Python path for local imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from image_loader import load_image
from food_detector import detect_food
from nutrition_matcher import match_nutrition
from result_formatter import summarize_nutrition


st.set_page_config(page_title="Image Nutrition Analyzer")
st.title("Image-Based Nutritional Analyzer")


@st.cache_data(show_spinner=False)
def _load_image_bytes(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_boxes(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    boxed = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(
            boxed,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return boxed


uploaded = st.sidebar.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
show_boxes = st.sidebar.checkbox("Show detection boxes", value=True)

if uploaded:
    image = _load_image_bytes(uploaded)
    detections = detect_food(image)
    labels = [d["label"] for d in detections]
    nutrition_df = match_nutrition(labels)
    summary_text, chart_bytes = summarize_nutrition(nutrition_df)

    if show_boxes:
        image = draw_boxes(image, detections)
    st.image(image, caption="Processed Image", use_column_width=True)

    st.subheader("Nutrition Summary")
    st.text(summary_text)
    st.image(chart_bytes)
else:
    st.info("Please upload an image to begin.")

