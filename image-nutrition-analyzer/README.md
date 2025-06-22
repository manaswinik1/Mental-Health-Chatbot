# Image-Based Nutritional Analyzer

This project estimates the nutritional content of foods from an input image. It
uses a YOLOv5 object detection model to identify food items and then matches
those items with a reference nutrition table.

## Features
- Upload an image of food
- YOLO-based food detection
- Nutrition lookup from a CSV table
- Streamlit dashboard interface

## Setup
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Folder Structure
```
image-nutrition-analyzer/
├── data/
│   └── raw/
│       ├── images/
│       └── nutrition_lookup.csv
├── models/
│   └── yolov5s.pt
├── src/
│   ├── image_loader.py
│   ├── food_detector.py
│   ├── nutrition_matcher.py
│   └── result_formatter.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── LICENSE
```

![screenshot placeholder](docs/screenshot.png)

## Disclaimer
This tool provides estimated nutrition information only. It is not intended for
medical or dietary advice.
