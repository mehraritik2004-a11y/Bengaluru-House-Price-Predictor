import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    with open(BASE_DIR / "house_price_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_locations():
    with open(BASE_DIR / "locations.json", "r") as f:
        return json.load(f)


pipe = load_model()
locations = load_locations()

st.title("🏠 Bengaluru House Price Predictor")
st.markdown("#### Predict house prices using Machine Learning!")
st.markdown("*BSc DS&AI · IIT Guwahati · DA378*")
st.divider()

location = st.selectbox("📍 Select Location", locations)

col1, col2 = st.columns(2)
with col1:
    bhk = st.selectbox("🛏 BHK", [1, 2, 3, 4, 5])
with col2:
    bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5])

total_sqft = st.slider("📐 Total Square Feet", min_value=300, max_value=8000, value=1200, step=100)

st.divider()

if st.button("🔮 Predict Price", use_container_width=True):
    input_df = pd.DataFrame([{
        "location": location,
        "total_sqft": total_sqft,
        "bath": float(bath),
        "bhk": bhk,
    }])

    predicted_price = pipe.predict(input_df)[0]
    price_per_sqft = (predicted_price * 100000) / total_sqft

    st.success(f"💰 Estimated Price: ₹ {predicted_price:.2f} Lakhs")
    st.info(f"📊 Price per Sqft: ₹ {price_per_sqft:,.0f}")

    st.markdown("---")
    st.markdown("**Your Input Summary:**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Location", location)
    c2.metric("BHK", bhk)
    c3.metric("Bathrooms", bath)
    c4.metric("Sqft", total_sqft)

st.divider()
st.caption("Bengaluru House Dataset · ML Pipeline · IIT Guwahati Project")
