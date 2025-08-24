import streamlit as st
import pandas as pd
import pickle

# Load model and feature columns
model = pickle.load(open("model.pkl", "rb"))
feature_cols = pd.read_csv("features.csv", header=None).squeeze("columns").tolist()


# Load dataset for dropdown values
df = pd.read_csv("data/used-car-sales-listings-dataset-2025/used_car_listings.csv")

# Streamlit app
st.title("🚗 Used Car Price Predictor")
st.write("Enter car details to predict its selling price.")

# Dropdowns
make = st.selectbox("Car Brand", sorted(df["make"].dropna().unique()))
model_name = st.selectbox("Car Model", sorted(df[df["make"] == make]["model"].dropna().unique()))

fuel_col = [c for c in df.columns if c.lower() == "fuel_type"]
fuel = st.selectbox("Fuel Type", sorted(df[fuel_col[0]].dropna().unique())) if fuel_col else st.selectbox("Fuel Type", [])

trans_col = [c for c in df.columns if "trans" in c.lower()]
transmission = st.selectbox("Transmission", sorted(df[trans_col[0]].dropna().unique())) if trans_col else st.selectbox("Transmission", [])

seller_col = [c for c in df.columns if "seller" in c.lower()]
seller = st.selectbox("Seller Type", sorted(df[seller_col[0]].dropna().unique())) if seller_col else st.selectbox("Seller Type", [])

# Numeric inputs
year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2020)
kms = st.number_input("Mileage (in km)", min_value=0, step=1000)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

# Predict button
if st.button("Predict Price"):
    # Prepare input row
    input_dict = {
        "make": make,
        "model": model_name,
        "Fuel_Type": fuel,
        "Transmission": transmission,
        "Seller_Type": seller,
        "Year": year,
        "mileage": kms,
        "seats": seats
    }
    input_df = pd.DataFrame([input_dict])

    # One-hot encode and align columns with training
    input_df = pd.get_dummies(input_df)
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    # Predict price
    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: {price:,.2f}")

