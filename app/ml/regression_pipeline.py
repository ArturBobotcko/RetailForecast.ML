from dataclasses import dataclass

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

from app.ml.features import (
    add_lag_features,
    build_future_lag_features,
    detect_optimal_lags,
)
from app.ml.models import build_model, resolve_model_algorithm
from models.Model import Model


@dataclass
class ForecastPipelineResult:
    validation_actual: np.ndarray
    validation_pred: np.ndarray
    forecast_pred: np.ndarray
    feature_columns: list[str]
    lag_periods: list[int]
    validation_horizon: int


def run_regression_forecast(
    model_request: Model,
    frame: pd.DataFrame,
    target_col: str,
    base_features: list[str],
    time_col: str,
    future_periods,
    forecast_horizon: int,
    validation_fraction: float,
) -> ForecastPipelineResult:
    frame = frame.sort_values(time_col).reset_index(drop=True).copy()
    lag_periods = detect_optimal_lags(
        frame[target_col],
        forecast_horizon=forecast_horizon,
    )

    frame, lag_cols = add_lag_features(frame, target_col, lag_periods)
    frame["trend"] = np.arange(len(frame)) / len(frame)
    if "Квартал" in frame.columns:
        frame["sin_q"] = np.sin(2 * np.pi * frame["Квартал"] / 4)
        frame["cos_q"] = np.cos(2 * np.pi * frame["Квартал"] / 4)

    future_x = pd.DataFrame(
        {
            feature: [frame[feature].iloc[-1]] * forecast_horizon
            for feature in base_features
        }
    )
    future_lag_df = build_future_lag_features(
        frame,
        target_col,
        lag_periods,
        forecast_horizon,
    )
    future_x = pd.concat([future_x.reset_index(drop=True), future_lag_df], axis=1)

    all_features = base_features + lag_cols
    split_data = split_and_scale_time_series_data(
        history_df=frame,
        future_df=future_x.assign(**{time_col: future_periods}),
        time_col=time_col,
        target_col=target_col,
        features=all_features,
        validation_fraction=validation_fraction,
    )

    _, validation_pred, forecast_pred = train_and_predict(
        model_request=model_request,
        split_data=split_data,
    )
    validation_actual = split_data["y_val"].to_numpy(dtype=float)

    return ForecastPipelineResult(
        validation_actual=validation_actual,
        validation_pred=np.asarray(validation_pred, dtype=float),
        forecast_pred=np.asarray(forecast_pred, dtype=float),
        feature_columns=all_features,
        lag_periods=lag_periods,
        validation_horizon=len(validation_actual),
    )


def split_and_scale_time_series_data(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    time_col: str,
    target_col: str,
    features: list[str],
    validation_fraction: float = 0.2,
    scaler: StandardScaler | None = None,
) -> dict[str, pd.DataFrame | pd.Series | StandardScaler]:
    history_df = history_df.sort_values(time_col).reset_index(drop=True).copy()
    future_df = future_df.sort_values(time_col).reset_index(drop=True).copy()

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    if len(history_df.index) < 2:
        raise ValueError(
            "Need at least two historical rows to create train and validation splits"
        )

    split_idx = int(len(history_df.index) * (1 - validation_fraction))
    split_idx = max(1, min(split_idx, len(history_df.index) - 1))

    train_df = history_df.iloc[:split_idx].copy()
    val_df = history_df.iloc[split_idx:].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split produced an empty subset")

    if scaler is None:
        scaler = StandardScaler()

    scaler.fit(train_df[features])

    x_train_scaled = pd.DataFrame(
        scaler.transform(train_df[features]),
        columns=features,
        index=train_df.index,
    )
    x_val_scaled = pd.DataFrame(
        scaler.transform(val_df[features]),
        columns=features,
        index=val_df.index,
    )
    x_future_scaled = pd.DataFrame(
        scaler.transform(future_df[features]),
        columns=features,
        index=future_df.index,
    )

    return {
        "X_train": train_df[features].copy(),
        "X_train_scaled": x_train_scaled,
        "y_train": train_df[target_col],
        "X_val": val_df[features].copy(),
        "X_val_scaled": x_val_scaled,
        "y_val": val_df[target_col],
        "X_future": future_df[features].copy(),
        "X_future_scaled": x_future_scaled,
        "future_times": future_df[time_col],
        "scaler": scaler,
        "train_df": train_df,
        "val_df": val_df,
        "future_df": future_df,
    }


def train_and_predict(
    model_request: Model,
    split_data: dict[str, pd.DataFrame | pd.Series | StandardScaler],
):
    algorithm = resolve_model_algorithm(model_request)

    if algorithm == "prophet":
        return _train_and_predict_prophet(split_data, split_data.get("future_times"))

    if algorithm == "arima":
        return _train_and_predict_arima(split_data)

    use_scaled_features = algorithm in {
        "lasso",
        "ridge",
        "elasticnet",
        "linear_regression",
    }

    feature_suffix = "_scaled" if use_scaled_features else ""
    x_train = split_data[f"X_train{feature_suffix}"]
    x_val = split_data[f"X_val{feature_suffix}"]
    x_future = split_data[f"X_future{feature_suffix}"]
    y_train = split_data["y_train"]

    model = build_model(algorithm, len(y_train))
    model.fit(x_train, y_train)

    validation_pred = model.predict(x_val)
    forecast_pred = model.predict(x_future)

    return model, validation_pred, forecast_pred


def _train_and_predict_prophet(split_data: dict, future_times) -> tuple:
    train_df = split_data["train_df"].copy()
    val_df = split_data["val_df"].copy()
    future_df = split_data["future_df"].copy()

    ds_col = "ds"
    train_df = train_df.rename(
        columns={
            "__period_index"
            if "__period_index" in train_df.columns
            else split_data.get("future_times", "Год"): ds_col
        }
    )
    val_df = val_df.rename(
        columns={
            "__period_index"
            if "__period_index" in val_df.columns
            else split_data.get("future_times", "Год"): ds_col
        }
    )
    future_df = future_df.rename(
        columns={
            "__period_index"
            if "__period_index" in future_df.columns
            else split_data.get("future_times", "Год"): ds_col
        }
    )

    train_df[ds_col] = pd.to_datetime(train_df[ds_col], errors="coerce")
    val_df[ds_col] = pd.to_datetime(val_df[ds_col], errors="coerce")
    future_df[ds_col] = pd.to_datetime(future_df[ds_col], errors="coerce")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )

    features = [
        column
        for column in split_data["X_train"].columns
        if column not in {"__period_index", "Год", "Квартал"}
    ]
    for feature in features:
        if feature in train_df.columns:
            model.add_regressor(feature)

    model.fit(train_df.rename(columns={split_data["y_train"].name: "y"}))

    validation_pred = model.predict(val_df)[["yhat"]].values.ravel()
    forecast_pred = model.predict(future_df)[["yhat"]].values.ravel()

    return model, validation_pred, forecast_pred


def _train_and_predict_arima(split_data: dict) -> tuple:
    val_df = split_data["val_df"].copy()
    future_df = split_data["future_df"].copy()

    y_train = split_data["y_train"].values
    exog_train = split_data["X_train"].values
    exog_val = split_data["X_val"].values
    exog_future = split_data["X_future"].values

    model = auto_arima(
        y_train,
        exogenous=exog_train,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        d=None,
        seasonal=True,
        m=4,
        start_P=0,
        start_Q=0,
        max_P=2,
        max_Q=2,
        D=None,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        n_fits=20,
    )

    validation_pred = model.predict(n_periods=len(val_df), exogenous=exog_val)
    forecast_pred = model.predict(n_periods=len(future_df), exogenous=exog_future)

    return model, validation_pred, forecast_pred
