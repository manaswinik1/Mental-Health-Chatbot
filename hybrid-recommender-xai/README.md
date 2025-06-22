# Explainable Hybrid Recommendation System

This project demonstrates a prototype hybrid recommender that combines
collaborative filtering and content-based techniques with sentence-transformer
embeddings. SHAP-based explanations describe why a particular item is
recommended.

## Features
- Personalized hybrid recommendations
- Embedding-based content similarity
- SHAP explanations for transparency
- Interactive Streamlit user interface

## Setup
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Folder Structure
```
hybrid-recommender-xai/
├── data/
│   └── raw/
│       ├── movielens_ratings.csv
│       └── movielens_metadata.csv
├── models/
├── src/
│   ├── data_loader.py
│   ├── collaborative_filtering.py
│   ├── content_embedder.py
│   ├── hybrid_recommender.py
│   └── explainer.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

![screenshot placeholder](docs/screenshot.png)

**Disclaimer**: This repository is a prototype meant for educational
purposes only and should not be used directly in production systems.
