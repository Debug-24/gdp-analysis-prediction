

import pathlib
import pandas as pd
from src.models.compare import compare_one_country

if __name__ == "__main__":

    in_csv = pathlib.Path("data/data-processed/clean_quarterly.csv")
    out_dir = pathlib.Path("data/data-processed")

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(in_csv, parse_dates=["quarter"])
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{in_csv}'")
        print("Please run the cleaner script first: python -m scripts.run_cleaner")
        exit()

    countries = sorted(df["country"].unique())

    rows = []
    print("Running model comparison for all countries...")
    for c in countries:
        res = compare_one_country(df, c)
        for label in ["Traditional", "Enhanced"]:
            if res.get(label):
                m = res[label]["metrics"]
                rows.append({
                    "country": c,
                    "feature_set": label,
                    "winner_model": res[label]["winner"],
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                    "R2": m["R2"],
                    "AdjR2": m["AdjR2"],
                })

                preds_path = out_dir / f"preds_{c}_{label}.csv"
                res[label]["preds"].to_csv(preds_path, index=False)

                if res[label].get("importance"):

                    importance_df = pd.DataFrame(res[label]["importance"].items(), columns=[
                                                 'feature', 'importance'])

                    importance_df = importance_df.sort_values(
                        by='importance', ascending=False)

                    importance_path = out_dir / f"importance_{c}_{label}.csv"
                    importance_df.to_csv(importance_path, index=False)

    summary_path = out_dir / "summary_metrics.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"\nModel comparison complete. Summary saved -> {summary_path}")
