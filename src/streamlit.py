"""Healthcare Prediction Pipeline - Streamlit App entrypoint.
"""

import os
import sys

import pandas as pd
import streamlit as st

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_ingestion import validate_dataset, set_target_column

st.set_page_config(page_title="Healthcare Prediction Pipeline", layout="wide")

RAW_DATA_DIR = "data/raw"



st.title("Healthcare Prediction Pipeline")
st.header("1. Dataset Upload")

source = st.radio("Choose a data source", ["Upload CSV", "Use sample dataset"], horizontal=True)

df = None

if source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    available = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    choice = st.selectbox("Select a sample dataset", available)
    if choice:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, choice))

if df is not None:
    is_valid, issues = validate_dataset(df)

    if not is_valid:
        st.error("Dataset has issues:")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
        st.dataframe(df.head(20), use_container_width=True)

        target_col = st.selectbox("Select the target column", df.columns)

        if st.button("Confirm dataset and target"):
            st.session_state.raw_df = df
            st.session_state.target_col = set_target_column(df, target_col)
            st.success(f"Target set to '{target_col}'. Continue via the sidebar.")

st.divider()

if "raw_df" in st.session_state:
    active_df = st.session_state.raw_df
    st.info(
        f"Active dataset: {active_df.shape[0]} rows x {active_df.shape[1]} columns "
        f"| Target: {st.session_state.target_col}"
    )
else:
    st.info("No dataset confirmed yet.")
