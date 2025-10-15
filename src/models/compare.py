

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from src.models.utils import ts_split_indices, metrics_dict, add_adj_r2
from sklearn.inspection import permutation_importance

TRADITIONAL = ["Consumption", "Investment",
               "TradeBalance", "inflation", "unemployment"]
ENHANCED = TRADITIONAL + ["WorkingAgePop", "DependencyRatio", "GDP_lag1"]


def train_and_eval(df_country: pd.DataFrame, features: list, model_name="Linear"):
    X = df_country[features].values
    y = df_country["GDP"].values
    n = len(df_country)
    if n < 16:
        return None

    train_idx, test_idx = ts_split_indices(n, test_size=8)
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    if model_name == "Linear":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    p_result = permutation_importance(
        model, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)
    importance_dict = dict(zip(features, p_result.importances_mean))

    mets = metrics_dict(yte, pred)
    mets = add_adj_r2(mets, n=len(yte), p=len(features))

    out = pd.DataFrame({
        "quarter": df_country.iloc[test_idx]["quarter"].values,
        "GDP_actual": yte,
        "GDP_pred": pred
    })

    return {"metrics": mets, "preds": out, "model": model, "importance": importance_dict}


def compare_one_country(df, country):
    g = df[df["country"] == country].dropna().copy()

    results = {}
    for label, feats in [("Traditional", TRADITIONAL), ("Enhanced", ENHANCED)]:
        feats = [f for f in feats if f in g.columns]
        res_lin = train_and_eval(g, feats, "Linear")
        res_ridge = train_and_eval(g, feats, "Ridge")
        res_rf = train_and_eval(g, feats, "RF")
        candidates = [("Linear", res_lin), ("Ridge", res_ridge),
                      ("RandomForest", res_rf)]
        candidates = [(n, r) for n, r in candidates if r is not None]
        if not candidates:
            results[label] = None
            continue

        best_name, best = sorted(
            candidates, key=lambda x: x[1]["metrics"]["RMSE"])[0]
        results[label] = {"winner": best_name, **best}
    return results
