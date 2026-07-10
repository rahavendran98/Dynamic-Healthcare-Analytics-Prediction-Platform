# Dynamic Healthcare Analytics & Prediction Platform

A Streamlit app and ML pipeline for healthcare prediction (diabetes, cardiovascular disease, stroke). Upload any CSV, pick a target column, and run it through cleaning, EDA, feature engineering, model training, and prediction.

## Requirements

- Python 
- See `requirements.txt` for package versions

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

## Pipeline

1. **Ingest** a CSV and select the target column
2. **Clean** missing values, duplicates, and outliers
3. **Explore** with automated EDA
4. **Engineer features** and select the most relevant ones
5. **Train & compare models**, then tune the best one
6. **Predict** on new data using the saved model
