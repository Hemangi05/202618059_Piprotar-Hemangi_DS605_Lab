import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Airbnb Price Prediction",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("airbnb_price_model.pkl")


model = load_model()


st.title("Airbnb Price Prediction")

st.write(
    "Enter the details of an Airbnb listing to estimate "
    "its nightly price."
)


st.sidebar.header("Listing Information")


neighbourhood_group = st.sidebar.selectbox(
    "Neighbourhood Group",
    [
        "Bronx",
        "Brooklyn",
        "Manhattan",
        "Queens",
        "Staten Island"
    ]
)


room_type = st.sidebar.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)


neighbourhood = st.sidebar.text_input(
    "Neighbourhood",
    "Midtown"
)


latitude = st.sidebar.number_input(
    "Latitude",
    value=40.75,
    format="%.6f"
)


longitude = st.sidebar.number_input(
    "Longitude",
    value=-73.98,
    format="%.6f"
)


minimum_nights = st.sidebar.number_input(
    "Minimum Nights",
    min_value=1,
    value=3
)


number_of_reviews = st.sidebar.number_input(
    "Number of Reviews",
    min_value=0,
    value=20
)


reviews_per_month = st.sidebar.number_input(
    "Reviews Per Month",
    min_value=0.0,
    value=1.0
)


calculated_host_listings_count = st.sidebar.number_input(
    "Host Listings Count",
    min_value=1,
    value=1
)


availability_365 = st.sidebar.number_input(
    "Availability in 365 Days",
    min_value=0,
    max_value=365,
    value=200
)


has_review = st.sidebar.selectbox(
    "Has Review",
    [0, 1]
)


days_since_last_review = st.sidebar.number_input(
    "Days Since Last Review",
    min_value=0,
    value=30
)


review_availability_ratio = (
    number_of_reviews /
    (availability_365 + 1)
)


minimum_nights_log = np.log1p(
    minimum_nights
)


input_data = pd.DataFrame({
    "neighbourhood_group": [neighbourhood_group],
    "neighbourhood": [neighbourhood],
    "latitude": [latitude],
    "longitude": [longitude],
    "room_type": [room_type],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews],
    "reviews_per_month": [reviews_per_month],
    "calculated_host_listings_count": [
        calculated_host_listings_count
    ],
    "availability_365": [availability_365],
    "has_review": [has_review],
    "days_since_last_review": [
        days_since_last_review
    ],
    "review_availability_ratio": [
        review_availability_ratio
    ],
    "minimum_nights_log": [
        minimum_nights_log
    ]
})


st.subheader("Listing Details")

st.dataframe(
    input_data,
    use_container_width=True
)


if st.button("Predict Price"):

    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success(
        f"Estimated Nightly Price: ${predicted_price:.2f}"
    )