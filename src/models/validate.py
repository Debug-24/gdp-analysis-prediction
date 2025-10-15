import pandas as pd
import pathlib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from src.models.utils import metrics_dict, add_adj_r2
from src.models.compare import TRADITIONAL, ENHANCED
import warnings

# suppress warnings for cleaner output
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
    #metrics = add_adj_r2(metrics, n=len(y_val), p=len(features))

    # put into data frame
    output_df = pd.DataFrame({
        "quarter": validation_df["quarter"],
        "GDP_actual": y_val,
        "GDP_pred": predictions
    })

    return {"metrics": metrics, "predictions": output_df, "model": model}

if __name__ == '__main__':
    # file paths
    processed_data_path = pathlib.Path("data/data-processed/clean_quarterly.csv")
    summary_metrics_path = pathlib.Path("data/data-processed/summary_metrics.csv")
    output_path = pathlib.Path("data/data-processed/validation_metrics.csv")
    output_dir = pathlib.Path("data/data-processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # load datasets
    try:
        full_data = pd.read_csv(processed_data_path, parse_dates=["quarter"])
        summary_df = pd.read_csv(summary_metrics_path)
    except FileNotFoundError as e:
        print(f"\nError: Could not find a required file: {e.filename}")
        exit()

    # --- CONFIGURATION ---
    VALIDATION_START_DATE = '2020-01-01'  # train on data before this, validate on data after

    validation_results = []

    # validate through each best model configuration from the summary file
    for index, row in summary_df.iterrows():
        country = row['country']
        feature_set_label = row['feature_set']
        winner_model = row['winner_model']

        #print(f"\nValidating for {country} | Feature Set: {feature_set_label} | Best Model: {winner_model}")

        # select the correct feature list based on the label
        features = ENHANCED if feature_set_label == 'Enhanced' else TRADITIONAL

        # filter data for the specific country and drop rows with missing values
        country_df = full_data[full_data['country'] == country].copy().dropna()
        
        # check for required features in the dataframe
        features_available = [f for f in features if f in country_df.columns]
        if not features_available:
            print("Skipping: Not all required features were found in the data for this country.")
            continue

        # validation with winning model on the specified time split
        validation_run_result = validate_model(country_df, features_available, winner_model, VALIDATION_START_DATE)

        if validation_run_result:
            val_metrics = validation_run_result['metrics']

            # dictionary to store comparison results
            result_row = {
                'country': country,
                'feature_set': feature_set_label,
                'winner_model': winner_model,
            }
            
            # compare original metrics from summary file with new validation metrics
            for metric in ['RMSE', 'MAE', 'R2', 'AdjR2']:
                original_metric = row[metric]
                validation_metric = val_metrics.get(metric)
                
                result_row[f'{metric}_original'] = original_metric
                result_row[f'{metric}_validation'] = validation_metric
                result_row[f'{metric}_diff'] = validation_metric - original_metric if validation_metric is not None else None

            validation_results.append(result_row)
            #print(f"Validation finished. RMSE Diff: {result_row['RMSE_diff']:.4f}")
        else:
            print("Could not complete validation for this configuration.")

    # create and save final validation metrics DataFrame
    if validation_results:
        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv(output_path, index=False)
        print(f"\nValidation complete. Results saved to {output_path}")
        print("Sample of validation results:")
        print(validation_df.head())
    else:
        print("\nWarning: No validation results were generated.")