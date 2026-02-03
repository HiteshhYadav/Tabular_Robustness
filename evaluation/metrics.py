from sklearn.metrics import (
    roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def classification_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred)
    }

def regression_metrics(y_true, y_pred):
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
