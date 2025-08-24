import pandas as pd

# Load dataset
file_path = r'used-car-sales-listings-dataset-2025\used_car_listings.csv'
df = pd.read_csv(file_path)

# Display basic info
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)
