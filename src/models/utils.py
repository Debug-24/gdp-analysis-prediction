
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def ts_split_indices(n: int, test_size=8) -> tuple:

    train_end = n - test_size
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(train_end, n)
    return train_idx, test_idx


def metrics_dict(y_true, y_pred) -> dict:

    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def add_adj_r2(metrics: dict, n: int, p: int) -> dict:
    """
    n = number of samples
    p = number of predictors (features)
    """
    r2 = metrics.get("R2", 0)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    metrics["AdjR2"] = adj_r2
    return metrics
