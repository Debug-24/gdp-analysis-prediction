

import pathlib
import pandas as pd
from src.data.clean import run

if __name__ == "__main__":

    raw_in = pathlib.Path("data/data-raw/gdp_dataset_for_ml.csv")
    out_dir = pathlib.Path("data/data-processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "clean_quarterly.csv"

    print("Loading raw data...")

    try:
        raw_df = pd.read_csv(raw_in)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at '{raw_in}'")
        print("Please ensure your gdp_dataset_for_ml.csv is in the data/data-raw folder.")
        exit()

    print(f"Original row count: {len(raw_df)}")

    print("Filtering data to include only records from 2005-01-01 onwards...")

    raw_df['quarter'] = pd.to_datetime(raw_df['quarter'])

    filtered_df = raw_df[raw_df['quarter'] >= '2005-01-01'].copy()

    print(f"Filtered row count: {len(filtered_df)}")

    print("\nStarting cleaning pipeline on the filtered data...")

    run(filtered_df, str(out_csv))

    print(f"\nFiltered and cleaned dataset saved -> {out_csv}")
