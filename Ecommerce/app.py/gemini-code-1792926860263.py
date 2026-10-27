import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Shipment Delivery Predictor",
    page_icon="📦",
    layout="centered"
)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# App Title & Description
st.title("📦 E-Commerce Shipment Delivery Predictor")
st.write("This web application predicts whether an e-commerce order will arrive **on time** or be **delayed** based on shipment and product details.")

st.divider()

# Sidebar / Form inputs for user data
st.header("Enter Shipment Details")

warehouse_block = st.selectbox("Warehouse Block", ('A', 'B', 'C', 'D', 'F'))
mode_of_shipment = st.selectbox("Mode of Shipment", ('Flight', 'Ship', 'Road'))
customer_care_calls = st.slider("Customer Care Calls", min_value=2, max_value=7, value=4, help="Number of calls made for inquiry of shipment")
customer_rating = st.slider("Customer Rating", min_value=1, max_value=5, value=3, help="Company rating from 1 (worst) to 5 (best)")
cost_of_the_product = st.number_input("Cost of the Product ($)", min_value=50, max_value=500, value=200)
prior_purchases = st.slider("Prior Purchases", min_value=2, max_value=10, value=3, help="Number of prior purchases")
product_importance = st.selectbox("Product Importance", ('low', 'medium', 'high'))
gender = st.selectbox("Gender", ('F', 'M'))
discount_offered = st.number_input("Discount Offered ($)", min_value=0, max_value=100, value=10)
weight_in_gms = st.number_input("Weight in grams", min_value=1000, max_value=8000, value=3000)

# Organize input into a DataFrame matching the training data structure
input_data = pd.DataFrame({
    'Warehouse_block': [warehouse_block],
    'Mode_of_Shipment': [mode_of_shipment],
    'Customer_care_calls': [customer_care_calls],
    'Customer_rating': [customer_rating],
    'Cost_of_the_Product': [cost_of_the_product],
    'Prior_purchases': [prior_purchases],
    'Product_importance': [product_importance],
    'Gender': [gender],
    'Discount_offered': [discount_offered],
    'Weight_in_gms': [weight_in_gms]
})

st.subheader("Selected Parameters Summary")
st.dataframe(input_data, use_container_width=True)

# Prediction Button
if st.button("Predict Delivery Status", type="primary", use_container_width=True):
    # Make prediction using the loaded pipeline model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.divider()
    st.subheader("Prediction Result")
    
    # Based on the dataset: 1 = Late/Delayed, 0 = Reached on Time
    if prediction[0] == 1:
        st.error("⚠️ **Warning: Delayed Shipment!** This product is predicted to **arrive late**.")
    else:
        st.success("✅ **On Time!** This product is predicted to **arrive on schedule**.")
        
    # Show confidence score
    confidence = prediction_proba[0][prediction[0]] * 100
    st.info(f"Model Confidence Score: **{confidence:.2f}%**")
