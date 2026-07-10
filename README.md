# Dynamic Healthcare Analytics & Prediction Platform

A Streamlit app and ML pipeline for healthcare prediction (diabetes, cardiovascular disease, stroke). Upload any CSV, pick a target column, and run it through cleaning, EDA, feature engineering, model training, and prediction.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Structure

- `app.py`, `pages/` – Streamlit app
- `src/` – pipeline modules (ingestion, cleaning, EDA, feature engineering, training, prediction)
- `notebooks/` – exploratory notebooks
- `data/` – raw and processed datasets
- `models/` – trained models and artifacts
- `reports/` – generated metrics and plots
