# Mental Health Chatbot (Offline AI Assistant)

This project implements a simple, offline-friendly mental health chatbot that uses a pre-downloaded dataset of therapeutic dialogues. It aims to provide empathetic responses while applying a basic safety filter to ensure that sensitive conversations remain supportive.

## Features
- Data loader for the EmpatheticDialogues dataset
- Response generator that matches user input to existing prompts
- Safety filter that removes potentially harmful content
- Streamlit interface for easy interaction

## Dataset
The chatbot relies on the [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) dataset, assumed to be located at `data/raw/empathetic_dialogues.csv`.

## Setup
1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Project Structure
```
mental-health-chatbot/
├── data/
│   └── raw/
│       └── empathetic_dialogues.csv  # not included in repository
├── models/
├── src/
│   ├── data_loader.py
│   ├── response_generator.py
│   └── safety_filter.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Disclaimer
This chatbot is meant as a simple demonstration and is **not** a substitute for professional mental health support. If you are struggling, please seek help from a qualified professional.

![screenshot placeholder](docs/screenshot.png)
