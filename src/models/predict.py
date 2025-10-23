import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

TRADITIONAL = ["Consumption", "Investment", "TradeBalance", "inflation", "unemployment"]
ENHANCED = TRADITIONAL + ["WorkingAgePop", "DependencyRatio", "GDP_lag1"]


def get_model_instance(model_name: str):
    if model_name == "Linear":
        return LinearRegression()
    elif model_name == "Ridge":
        return Ridge(alpha=1.0)
    elif model_name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        raise ValueError(f"Unknown model name provided: {model_name}")


def generate_forecast(model_name: str, country_df: pd.DataFrame, feature_set_label: str, future_quarters: int):
    features = ENHANCED if feature_set_label == 'Enhanced' else TRADITIONAL
    features = [f for f in features if f in country_df.columns]
    
    model = get_model_instance(model_name)
    X_full = country_df[features]
    y_full = country_df["GDP"]
    model.fit(X_full, y_full)
    
    last_known_features = X_full.iloc[-1].to_dict()
    last_known_gdp = y_full.iloc[-1]
    last_known_date = country_df['quarter'].max()

    predictions = []
    if 'GDP_lag1' in last_known_features:
        last_known_features['GDP_lag1'] = last_known_gdp
        
    for _ in range(future_quarters):
        current_features_df = pd.DataFrame([last_known_features])
        next_prediction = model.predict(current_features_df)[0]
        predictions.append(next_prediction)
        if 'GDP_lag1' in last_known_features:
            last_known_features['GDP_lag1'] = next_prediction

    future_dates = pd.date_range(start=last_known_date, periods=future_quarters + 1, freq='QS-OCT')[1:]
    forecast_df = pd.DataFrame({
        'quarter': future_dates,
        'GDP_pred': predictions
    })
    
    return forecast_df
