

import pandas as pd

KEEP_COLS = [
    "country",
    "quarter",
    "gross_fixed_capital_formation",
    "final_consumption_expenditure",
    "gross_domestic_product",
    "external_balance_of_goods_and_services",
    "inflation",
    "unemployment",
    "working_age_population",
    "dependency_ratio",
]

RENAME = {
    "gross_domestic_product": "GDP",
    "final_consumption_expenditure": "Consumption",
    "gross_fixed_capital_formation": "Investment",
    "external_balance_of_goods_and_services": "TradeBalance",
    "working_age_population": "WorkingAgePop",
    "dependency_ratio": "DependencyRatio",
}


def clean_types(df: pd.DataFrame) -> pd.DataFrame:

    df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
    df = df.dropna(subset=["quarter"])
    for c in set(df.columns) - {"country", "quarter"}:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_names(df: pd.DataFrame) -> pd.DataFrame:

    return df.rename(columns=RENAME)


def dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["country", "quarter"]).copy()
    df = df.sort_values(["country", "quarter"]).reset_index(drop=True)
    return df


def impute_basic(df: pd.DataFrame) -> pd.DataFrame:

    def _fix(g):
        g = g.copy()
        cols = [c for c in g.columns if c not in ("country", "quarter")]
        g[cols] = g[cols].ffill().bfill()
        return g
    return df.groupby("country", group_keys=False).apply(_fix)


def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["quarter"].dt.year
    df["q"] = df["quarter"].dt.quarter
    df["GDP_lag1"] = df.groupby("country")["GDP"].shift(1)
    return df


def run(df_in: pd.DataFrame, path_out: str):

    df = df_in[KEEP_COLS].copy()
    df.columns = [c.strip().lower() for c in df.columns]

    df = clean_types(df)
    df = normalize_names(df)
    df = dedupe_and_sort(df)
    df = impute_basic(df)
    df = add_time_feats(df)

    df = df.dropna(subset=["GDP"]).reset_index(drop=True)

    df.to_csv(path_out, index=False)
