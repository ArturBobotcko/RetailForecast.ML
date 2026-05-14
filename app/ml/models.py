import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from xgboost import XGBRegressor

from models.Model import Model


def resolve_model_algorithm(model_request: Model) -> str:
    raw_value = (
        (model_request.algorithm or model_request.name or "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "linear": "linear_regression",
        "linearregression": "linear_regression",
        "linear_regression": "linear_regression",
        "lasso": "lasso",
        "lassocv": "lasso",
        "ridge": "ridge",
        "ridgecv": "ridge",
        "elasticnet": "elasticnet",
        "elastic_net": "elasticnet",
        "elasticnetcv": "elasticnet",
        "xgboost": "xgboost",
        "prophet": "prophet",
        "arima": "arima",
        "auto_arima": "arima",
    }

    if raw_value not in aliases:
        raise ValueError(
            f"Unsupported model algorithm '{model_request.algorithm or model_request.name}'. "
            "Supported values: linear_regression, lasso, ridge, elasticnet"
        )

    return aliases[raw_value]


def build_model(algorithm: str, train_size: int):
    if train_size < 10:
        cv_folds = 2
    elif train_size < 50:
        cv_folds = 3
    else:
        cv_folds = min(5, train_size // 5)

    if algorithm == "linear_regression":
        return LinearRegression()

    if algorithm == "lasso":
        alphas = np.logspace(-3, 2, 12) if train_size > 50 else np.logspace(-2, 1, 8)
        return LassoCV(
            alphas=alphas,
            cv=cv_folds,
            max_iter=50000,
            tol=1e-4,
            random_state=42,
        )

    if algorithm == "ridge":
        alphas = np.logspace(-3, 2, 12) if train_size > 50 else np.logspace(-2, 1, 8)
        return RidgeCV(alphas=alphas, cv=cv_folds)

    if algorithm == "elasticnet":
        alphas = np.logspace(-3, 2, 12) if train_size > 50 else np.logspace(-2, 1, 8)
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
        return ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=cv_folds,
            max_iter=50000,
            tol=1e-4,
            random_state=42,
        )

    if algorithm == "xgboost":
        if train_size < 50:
            n_estimators = 50
            learning_rate = 0.2
            max_depth = 2
            min_child_weight = 5
            subsample = 0.6
            colsample_bytree = 0.6
        elif train_size < 100:
            n_estimators = 100
            learning_rate = 0.1
            max_depth = 3
            min_child_weight = 3
            subsample = 0.7
            colsample_bytree = 0.7
        else:
            n_estimators = 300
            learning_rate = 0.03
            max_depth = 5
            min_child_weight = 3
            subsample = 0.8
            colsample_bytree = 0.8

        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=0.6 if train_size < 50 else 0.8,
            reg_alpha=0.1 if train_size < 50 else 0.01,
            reg_lambda=2.0 if train_size < 50 else 1.0,
            random_state=42,
            objective="reg:squarederror",
            early_stopping_rounds=5 if train_size > 50 else None,
        )

    if algorithm == "prophet":
        return "prophet"
    if algorithm == "arima":
        return "arima"

    raise ValueError(f"Unsupported model algorithm '{algorithm}'")
