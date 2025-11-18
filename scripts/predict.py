import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# These lists define the two feature sets our models can use.
TRADITIONAL = ["Consumption", "Investment",
               "TradeBalance", "inflation", "unemployment"]
ENHANCED = TRADITIONAL + ["WorkingAgePop", "DependencyRatio", "GDP_lag1"]


def get_model_instance(model_name: str):
    # This helper function creates a new, ready-to-train model based on a name string.
    # This keeps our main function clean and ensures model settings are consistent.
    if model_name == "Linear":
        return LinearRegression()
    elif model_name == "Ridge":
        return Ridge(alpha=1.0)
    elif model_name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        # If the model name isn't recognized, we raise an error to stop the program.
        raise ValueError(f"Unknown model name provided: {model_name}")


def generate_forecast(model_name: str, country_df: pd.DataFrame, feature_set_label: str, future_quarters: int):
    """
    Takes the best model for a country, retrains it on all historical data,
    and then predicts future GDP values.

    Args:
        model_name: The name of the winning model (e.g., 'RandomForest').
        country_df: A DataFrame with the complete historical data for one country.
        feature_set_label: The name of the feature set to use ('Traditional' or 'Enhanced').
        future_quarters: How many quarters into the future we want to predict.

    Returns:
        A pandas DataFrame with the future dates and the corresponding GDP predictions.
    """

    # First, we set up the model and the features it will use.
    features = ENHANCED if feature_set_label == 'Enhanced' else TRADITIONAL
    # We double-check to only use features that are actually in this country's dataset.
    features = [f for f in features if f in country_df.columns]

    # Create a fresh instance of the winning model.
    model = get_model_instance(model_name)

    # We retrain the model on the *entire* historical dataset. This makes our
    # forecast as accurate as possible by giving the model all available information.
    X_full = country_df[features]
    y_full = country_df["GDP"]
    model.fit(X_full, y_full)

    # Now we start the forecasting process. We predict one quarter at a time in a loop.
    # To begin, we need the most recent real data to make our first prediction.
    last_known_features = X_full.iloc[-1].to_dict()
    last_known_gdp = y_full.iloc[-1]
    last_known_date = country_df['quarter'].max()

    predictions = []
    # Before the loop, we set the 'GDP_lag1' feature to the last *real* GDP value.
    if 'GDP_lag1' in last_known_features:
        last_known_features['GDP_lag1'] = last_known_gdp

    for _ in range(future_quarters):
        # We wrap the current set of features in a DataFrame to feed it to the model.
        current_features_df = pd.DataFrame([last_known_features])

        # Predict the GDP for the next quarter.
        next_prediction = model.predict(current_features_df)[0]
        predictions.append(next_prediction)

        # This is the key to the forecast: the prediction we just made
        # becomes the 'GDP_lag1' feature for the *next* quarter's prediction.
        if 'GDP_lag1' in last_known_features:
            last_known_features['GDP_lag1'] = next_prediction

    # Once the loop is done, we format our predictions into a clean DataFrame.
    # First, create a list of future dates for our forecast.
    future_dates = pd.date_range(
        start=last_known_date, periods=future_quarters + 1, freq='QS-OCT')[1:]

    # Then, combine the future dates with our list of predictions.
    forecast_df = pd.DataFrame({
        'quarter': future_dates,
        'GDP_pred': predictions
    })

    return forecast_df
