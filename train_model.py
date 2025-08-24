# train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

DATA_PATHS = [
    Path("data/used-car-sales-listings-dataset-2025/used_car_listings.csv"),
    Path("data/used_car_listings.csv"),
]

def load_data():
    for p in DATA_PATHS:
        if p.exists():
            print(f"Using dataset: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError("No dataset found in data/ folder.")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unneeded columns
    drop_cols = ["listing_id", "vin", "features", "location"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Rename target
    if "price" in df.columns:
        df.rename(columns={"price": "Selling_Price"}, inplace=True)

    # Handle missing values
    df = df.dropna(subset=["Selling_Price"])
    df.fillna({"trim": "Unknown", "body_type": "Unknown", "condition": "Unknown"}, inplace=True)
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce").fillna(df["mileage"].median())

    # Feature engineering
    df["car_age"] = 2025 - df["year"]
    df.drop(columns=["year"], inplace=True)

    # One-hot encode categoricals
    cat_cols = ["make", "model", "trim", "body_type", "fuel_type", "transmission", "condition", "seller_type"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

def main():
    df = load_data()
    df = preprocess(df)

    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")

    # Save model and feature names
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    pd.Series(X_train.columns).to_csv("features.csv", index=False, header=False)
    print("Model saved to model.pkl and features.csv")

if __name__ == "__main__":
    main()
