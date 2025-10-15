import pandas as pd
import pathlib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from src.models.utils import metrics_dict, add_adj_r2
from src.models.compare import TRADITIONAL, ENHANCED
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

def validate_model(df_country: pd.DataFrame, features: list, model_name: str, validation_start_date: str):
    """
    Trains a model on historical data and validates it on a subsequent, fixed time period.
    """
    # format for quarter column
    df_country['quarter'] = pd.to_datetime(df_country['quarter'])

    # data split
    train_df = df_country[df_country['quarter'] < validation_start_date].copy()
    validation_df = df_country[df_country['quarter'] >= validation_start_date].copy()

    # check if dataframes have enough data
    if train_df.empty or validation_df.empty:
        print(f"Warning: Insufficient data for training or validation with start date {validation_start_date}.")
        return None

    # define features and labels
    X_train = train_df[features].values
    y_train = train_df["GDP"].values
    X_val = validation_df[features].values
    y_val = validation_df["GDP"].values

    # select the specified model
    if model_name == "Linear":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=1.0)
    else:  # default to RandomForest
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    # train on the historical data
    model.fit(X_train, y_train)

    # predictions for the validation period
    predictions = model.predict(X_val)

    # calculate performance metrics
    metrics = metrics_dict(y_val, predictions)
    metrics = add_adj_r2(metrics, n=len(y_val), p=len(features))

    # put into data frame
    output_df = pd.DataFrame({
        "quarter": validation_df["quarter"],
        "GDP_actual": y_val,
        "GDP_pred": predictions
    })

    return {"metrics": metrics, "predictions": output_df, "model": model}


def run_validation_for_country(df: pd.DataFrame, country: str, validation_start_date: str):
    # filter data for the specified country and drop rows with missing values
    country_df = df[df["country"] == country].dropna().copy()

    if country_df.empty:
        print(f"No data available for country: {country}")
        return

    print(f"Running Validation for {country} (Validation Period >= {validation_start_date})")

    # go through both traditional and enhanced feature sets
    for feature_set_label, feature_list in [("Traditional", TRADITIONAL), ("Enhanced", ENHANCED)]:
        print(f"\n Set: {feature_set_label}")
        
        # check that features exist
        features_available = [f for f in feature_list if f in country_df.columns]
        
        if not features_available:
            print("Skipping: None of the required features were found in the data.")
            continue

        # validate each model type with the current feature set
        for model_name in ["Linear", "Ridge", "RF"]:
            print(f"\nValidating Model: {model_name}")
            result = validate_model(country_df.copy(), features_available, model_name, validation_start_date)
            
            if result:
                print("Performance Metrics:")
                for key, value in result["metrics"].items():
                    print(f"  {key}: {value:.4f}")
                
                print("\nSample Predictions:")
                print(result["predictions"].head())
            else:
                print(f"Could not complete validation for {model_name} model.")


if __name__ == '__main__':
    try:
        # load the processed dataset from the data pipeline
        processed_data_path = pathlib.Path("data/data-processed/clean_quarterly.csv")
        full_data = pd.read_csv(processed_data_path, parse_dates=["quarter"])


        # --- CONFIGURATION ---
        COUNTRY_TO_VALIDATE = 'United States'
        VALIDATION_START_DATE = '2020-01-01' # train on data before this, validate on data after

        # run the validation process
        run_validation_for_country(full_data, COUNTRY_TO_VALIDATE, VALIDATION_START_DATE)

    except FileNotFoundError:
        print(f"\nError: Processed data file not found at '{processed_data_path}'")