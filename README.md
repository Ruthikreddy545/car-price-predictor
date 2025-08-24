
# Car Price Predictor 🚗

End-to-end project to predict used car selling prices with a clean ML pipeline and a Streamlit app.

## Project Structure
```
car-price-predictor/
├── app.py
├── fetch_data.py
├── train_model.py
├── requirements.txt
├── data/
│   └── sample_car_data.csv
└── models/ (auto-created after training)
    ├── car_price_model.pkl
    └── model_features.csv
```

## Quick Start

1. **Create & activate a virtual environment (optional but recommended).**  
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the dataset (two options):**
   - **Option A (Recommended):** Use Kaggle API to download the full dataset.  
     ```bash
     pip install kaggle
     # Place kaggle.json (API token) under:
     # Windows: %USERPROFILE%\.kaggle\kaggle.json
     # macOS/Linux: ~/.kaggle/kaggle.json
     python fetch_data.py
     ```
     This will create `data/car_data.csv` from the "Vehicle Dataset from CarDekho".

   - **Option B:** Use the provided `data/sample_car_data.csv` (smaller) to try things out quickly.

4. **Train the model**  
   ```bash
   python train_model.py
   ```

5. **Run the app**  
   ```bash
   streamlit run app.py
   ```

The app will guide you to train the model if artifacts are missing.

## Notes
- Model: RandomForestRegressor with sensible defaults (can be tuned further).
- Features: Present_Price, Kms_Driven, Owner, Car_Age + one-hot encodings for Fuel, Seller, Transmission.
- Target unit: **lakhs** of INR (as per the dataset convention).

## Next Steps
- Add hyperparameter tuning (Grid/RandomizedSearch).
- Try Gradient Boosting (XGBoost / LightGBM) and compare.
- Log experiments with MLflow, add tests, and containerize for deployment.
