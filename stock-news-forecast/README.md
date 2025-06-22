# Stock Forecasting with News Fusion

This project combines time-series modeling of stock prices with sentiment analysis of financial news headlines. It demonstrates a basic approach for multimodal forecasting and includes an interactive dashboard built with Streamlit.

## Features
- Load historical stock prices and news headlines
- Sentiment scoring using FinBERT
- Feature engineering with technical indicators and aggregated sentiment
- Forecasting model using machine learning
- Interactive dashboard for exploring predictions

## Dataset Sources
- [Yahoo Finance](https://finance.yahoo.com/)
- [NewsAPI](https://newsapi.org/) or similar
- [Kaggle](https://www.kaggle.com/) for curated datasets

## Setup
```bash
pip install -r requirements.txt
streamlit run app/streamlit_dashboard.py
```

## Folder Structure
```
stock-news-forecast/
├── data/
│   └── raw/
│       ├── stock_prices.csv
│       └── news_headlines.csv
├── models/
├── src/
│   ├── data_loader.py
│   ├── sentiment_analyzer.py
│   ├── feature_engineering.py
│   ├── forecasting_model.py
├── app/
│   └── streamlit_dashboard.py
├── requirements.txt
├── README.md
└── LICENSE
```

![Dashboard Screenshot](docs/screenshot_placeholder.png)

> **Disclaimer**: This project is for educational purposes only and should not be used as financial advice.
