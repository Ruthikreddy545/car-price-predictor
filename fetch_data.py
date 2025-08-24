
import os
import zipfile
from pathlib import Path

# This script uses Kaggle CLI. Steps:
# 1) pip install kaggle
# 2) Place your Kaggle API token at ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)
# 3) Run: python fetch_data.py
#
# It will download 'vehicle-dataset-from-cardekho' and place car_data.csv under data/

def main():
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    # Download via Kaggle CLI
    exit_code = os.system("kaggle datasets download -d nehalbirla/vehicle-dataset-from-cardekho -p \"%s\"" % data_dir.as_posix())
    if exit_code != 0:
        print("Failed to download via Kaggle CLI. Please ensure Kaggle is installed and API token is configured.")
        return

    # Find and extract the downloaded zip
    zips = list(data_dir.glob("*.zip"))
    if not zips:
        print("Dataset zip not found after download.")
        return

    with zipfile.ZipFile(zips[0], 'r') as zf:
        zf.extractall(data_dir)

    # Rename to a consistent filename if needed
    # The Kaggle dataset typically contains 'CAR_DETAILS_FROM_CAR_DEKHO.csv'
    candidates = [
        data_dir / "car data.csv",
        data_dir / "CAR DETAILS FROM CAR DEKHO.csv",
        data_dir / "car_data.csv",
        data_dir / "CAR_DETAILS_FROM_CAR_DEKHO.csv",
    ]
    final = data_dir / "car_data.csv"
    for c in candidates:
        if c.exists():
            c.rename(final)
            break

    print("Downloaded & prepared:", final)

if __name__ == "__main__":
    main()
