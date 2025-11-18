import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "")

st.markdown("# Iris prediction")

st.markdown("## Please provide values")
col1, col2, col3, col4 = st.columns(4)
with col1:
    sepal_length = st.number_input("Sepal length (cm)")
with col2:
    sepal_width = st.number_input("Sepal width (cm)")
with col3:
    petal_length = st.number_input("Petal length (cm)")
with col4:
    petal_width = st.number_input("Petal width (cm)")


def fetch_iris_prediction(
    sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm
):
    if not API_URL:
        raise Exception("API_URL not found")
    response = requests.post(
        API_URL,
        data=json.dumps(
            {
                "sepal_length_cm": sepal_length_cm,
                "sepal_width_cm": sepal_width_cm,
                "petal_length_cm": petal_length_cm,
                "petal_width_cm": petal_width_cm,
            }
        ),
    )
    return response.json()


if st.button("Predict"):
    st.markdown(
        f"Predicted class: {fetch_iris_prediction(sepal_length, sepal_width, petal_length, petal_width)['prediction']}"
    )
