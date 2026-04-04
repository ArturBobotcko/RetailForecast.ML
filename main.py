import csv
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import traceback

import httpx
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from DTOs.TrainingRunRequest import TrainingRunRequest
from DTOs.TrainingRunCallbackRequest import TrainingRunCallbackRequest
from DTOs.TrainingRunResponse import TrainingRunResponse
from models.Model import Model
from models.Metric import Metric
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.stattools import acf

app = FastAPI()
DEFAULT_VALIDATION_FRACTION = 0.2
GLOBAL_RANDOM_SEED = 42


def set_all_seeds(seed: int = GLOBAL_RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


set_all_seeds(GLOBAL_RANDOM_SEED)


@app.post("/api/trainingrun/start", response_model=TrainingRunResponse)
async def training_run_start(request: TrainingRunRequest, background_tasks: BackgroundTasks):
    external_job_id = str(uuid4())
    background_tasks.add_task(process_training_run, request, external_job_id)

    return TrainingRunResponse(
        externalJobId=external_job_id,
        status="Running",
        message=f"Training started for run {request.trainingRunId}",
    )


async def process_training_run(request: TrainingRunRequest, external_job_id: str):
    temp_file_path = None

    try:
        temp_file_path = await download_dataset(request.downloadUrl)
        data = load_data(temp_file_path)

        time_col = request.timeColumn.strip() if request.timeColumn else "Год"
        target_col = request.targetColumn.strip()
        forecast_frequency = normalize_forecast_frequency(request.forecastFrequency)

        if target_col not in data.columns:
            raise ValueError(f"Missing column '{target_col}'")

        if not pd.api.types.is_numeric_dtype(pd.to_numeric(data[target_col], errors="coerce")):
            raise ValueError(f"Column '{target_col}' must be numeric")

        reserved_time_columns = {time_col, target_col}
        if forecast_frequency in {"Auto", "Quarterly"}:
            reserved_time_columns.update({"Год", "Квартал"})

        features = [
            column for column in request.featureColumns
            if column in data.columns
            and column not in reserved_time_columns
            and pd.api.types.is_numeric_dtype(pd.to_numeric(data[column], errors="coerce"))
        ]

        if not features:
            features = [
                column for column in data.columns
                if column not in reserved_time_columns
                and pd.api.types.is_numeric_dtype(pd.to_numeric(data[column], errors="coerce"))
            ]

        if not features:
            raise ValueError("No numeric feature columns found")

        forecast_horizon = request.forecastHorizon or 1
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be greater than 0")

        resolved_frequency = detect_forecast_frequency(data, forecast_frequency)

        if resolved_frequency == "Quarterly":
            frame, future_periods, forecast_timestamps = prepare_quarterly_training_data(
                data,
                target_col,
                features,
                forecast_horizon,
            )
            time_values = frame["__period_index"].to_numpy(dtype=float).reshape(-1, 1)
            split_time_col = "__period_index"
        else:
            frame, future_periods, forecast_timestamps, resolved_time_col = prepare_yearly_training_data(
                data,
                time_col,
                target_col,
                features,
                forecast_horizon,
            )
            time_values = frame[resolved_time_col].to_numpy(dtype=float).reshape(-1, 1)
            split_time_col = resolved_time_col

        LAG_PERIODS = detect_optimal_lags(frame[target_col], forecast_horizon=forecast_horizon)
        frame["Q1"] = (frame["Квартал"] == 1).astype(int)
        frame["Q2"] = (frame["Квартал"] == 2).astype(int)
        frame["Q3"] = (frame["Квартал"] == 3).astype(int)

        frame, lag_cols = add_lag_features(frame, target_col, LAG_PERIODS)
        time_values = frame[split_time_col].to_numpy(dtype=float).reshape(-1, 1)

        # Feature extrapolation disabled for now - focus on training features
        feature_models = {}
        for feature in features:
            # Use last value forward for simple extrapolation
            last_value = frame[feature].iloc[-1]
            # Create a simple model that predicts last value
            class ConstantPredictor:
                def __init__(self, value):
                    self.value = value
                def predict(self, X):
                    return np.full(len(X), self.value)
            feature_models[feature] = ConstantPredictor(last_value)

        future_x = pd.DataFrame({
            feature: feature_models[feature].predict(future_periods.reshape(-1, 1))
            for feature in features
        })

        future_lag_df = build_future_lag_features(frame, target_col, LAG_PERIODS, forecast_horizon)
        future_x = pd.concat([future_x.reset_index(drop=True), future_lag_df], axis=1)

        all_features = features + lag_cols

        split_data = split_and_scale_time_series_data(
            history_df=frame,
            future_df=future_x.assign(**{split_time_col: future_periods}),
            time_col=split_time_col,
            target_col=target_col,
            features=all_features,
            validation_fraction=DEFAULT_VALIDATION_FRACTION,
        )

        _, validation_pred, forecast_pred = train_and_predict(
            model_request=request.model,
            split_data=split_data,
        )

        actual = split_data["y_val"].to_numpy(dtype=float)
        
        mae = float(np.mean(np.abs(actual - validation_pred)))
        rmse = float(np.sqrt(np.mean((actual - validation_pred) ** 2)))
        ss_res = float(np.sum((actual - validation_pred) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 if ss_tot == 0 else float(1 - (ss_res / ss_tot))
        
        print(f"  ss_res: {ss_res:.2f}, ss_tot: {ss_tot:.2f}, R²: {r2:.4f}")
        
        # Robust MAPE calculation (handles near-zero values)
        mape_threshold = 1e-6
        mape_mask = np.abs(actual) > mape_threshold
        if mape_mask.sum() > 0:
            mape = float(np.mean(np.abs((actual[mape_mask] - validation_pred[mape_mask]) / actual[mape_mask])) * 100)
            mape = min(mape, 1000.0)  # Cap at 1000% to avoid extreme values
        else:
            mape = 0.0

        callback_request = TrainingRunCallbackRequest(
            status="Completed",
            metrics=[
                Metric(name="mae", value=round(mae, 6)),
                Metric(name="rmse", value=round(rmse, 6)),
                Metric(name="r2", value=round(r2, 6)),
                Metric(name="mape", value=round(mape, 2))
            ],
            forecast=[
                {
                    "timestamp": timestamp.isoformat(),
                    "value": round(float(value), 4),
                }
                for timestamp, value in zip(forecast_timestamps, forecast_pred, strict=True)
            ],
            error=None,
            externalJobId=external_job_id,
        )
    except Exception as ex:  # noqa: BLE001
        callback_request = TrainingRunCallbackRequest(
            status="Failed",
            metrics=[],
            forecast=[],
            error=f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}",
            externalJobId=external_job_id,
        )
    finally:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)

    await send_callback(request.callbackUrl, callback_request)


async def download_dataset(download_url: str) -> Path:
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        response = await client.get(download_url, follow_redirects=True)
        response.raise_for_status()

    suffix = ".csv"
    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition.lower():
        file_name = content_disposition.split("filename=", maxsplit=1)[1].strip().strip('"')
        resolved_suffix = Path(file_name).suffix.lower()
        if resolved_suffix in {".csv", ".xls", ".xlsx"}:
            suffix = resolved_suffix

    if suffix == ".csv":
        content_type = response.headers.get("content-type", "").lower()
        if "spreadsheetml" in content_type:
            suffix = ".xlsx"
        elif "excel" in content_type:
            suffix = ".xls"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(response.content)
        return Path(temp_file.name)


def load_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            sample = csv_file.read(4096)
            csv_file.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

        return pd.read_csv(file_path, delimiter=delimiter)

    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file format: {suffix}")

# Убирает строки с пустыми значениями в времени/целевой, сортирует, интерполирует пропуски в features
def prepare_training_frame(
    frame: pd.DataFrame,
    time_col: str,
    target_col: str,
    features: list[str]
) -> pd.DataFrame:
    frame = frame.dropna(subset=[time_col, target_col]).copy()
    if frame.empty:
        raise ValueError("Dataset does not contain enough rows after removing empty time/target values")

    frame = frame.sort_values(time_col).reset_index(drop=True)

    for feature in features:
        if frame[feature].isna().all():
            raise ValueError(f"Column '{feature}' contains only empty values")

        frame[feature] = frame[feature].interpolate(method="linear", limit_direction="both")
        frame[feature] = frame[feature].ffill().bfill()

    frame = frame.dropna(subset=features).copy()
    if len(frame.index) < 2:
        raise ValueError("Dataset must contain at least two valid rows after filling empty values")

    return frame


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
        raise ValueError("Need at least two historical rows to create train and validation splits")

    split_idx = int(len(history_df.index) * (1 - validation_fraction))
    split_idx = max(1, min(split_idx, len(history_df.index) - 1))

    train_df = history_df.iloc[:split_idx].copy()
    val_df = history_df.iloc[split_idx:].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split produced an empty subset")

    if scaler is None:
        scaler = StandardScaler()

    # FIT scaler on ALL historical data to ensure validation values are properly scaled
    scaler.fit(history_df[features])

    X_train_scaled = pd.DataFrame(
        scaler.transform(train_df[features]),
        columns=features,
        index=train_df.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(val_df[features]),
        columns=features,
        index=val_df.index,
    )
    X_future_scaled = pd.DataFrame(
        scaler.transform(future_df[features]),
        columns=features,
        index=future_df.index,
    )

    return {
        "X_train": train_df[features].copy(),
        "X_train_scaled": X_train_scaled,
        "y_train": train_df[target_col],
        "X_val": val_df[features].copy(),
        "X_val_scaled": X_val_scaled,
        "y_val": val_df[target_col],
        "X_future": future_df[features].copy(),
        "X_future_scaled": X_future_scaled,
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

    use_scaled_features = algorithm in {"lasso", "ridge", "elasticnet", "linear_regression", "xgboost"}

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
    """Минимальная обёртка для Prophet — не трогает остальную логику"""
    train_df = split_data["train_df"].copy()
    val_df = split_data["val_df"].copy()
    future_df = split_data["future_df"].copy()

    # Prophet требует колонку "ds" (datetime) и "y"
    ds_col = "ds"
    train_df = train_df.rename(columns={"__period_index" if "__period_index" in train_df.columns else split_data.get("future_times", "Год"): ds_col})
    val_df = val_df.rename(columns={"__period_index" if "__period_index" in val_df.columns else split_data.get("future_times", "Год"): ds_col})
    future_df = future_df.rename(columns={"__period_index" if "__period_index" in future_df.columns else split_data.get("future_times", "Год"): ds_col})

    # Преобразуем в datetime (Prophet любит настоящий datetime)
    train_df[ds_col] = pd.to_datetime(train_df[ds_col], errors="coerce")
    val_df[ds_col] = pd.to_datetime(val_df[ds_col], errors="coerce")
    future_df[ds_col] = pd.to_datetime(future_df[ds_col], errors="coerce")

    # Основная модель Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative"   # для ритейла обычно лучше
    )

    # Добавляем все фичи как регрессоры
    features = [col for col in split_data["X_train"].columns if col not in {"__period_index", "Год", "Квартал"}]
    for f in features:
        if f in train_df.columns:
            model.add_regressor(f)

    # Fit
    model.fit(train_df.rename(columns={split_data["y_train"].name: "y"}))

    # Predict на валидации и будущем
    val_pred = model.predict(val_df)[["yhat"]].values.ravel()
    forecast_pred = model.predict(future_df)[["yhat"]].values.ravel()

    return model, val_pred, forecast_pred

def _train_and_predict_arima(split_data: dict) -> tuple:
    """Auto-ARIMA (SARIMAX) с автоматическим подбором параметров"""
    train_df = split_data["train_df"].copy()
    val_df = split_data["val_df"].copy()
    future_df = split_data["future_df"].copy()

    y_train = split_data["y_train"].values
    exog_train = split_data["X_train"].values
    exog_val = split_data["X_val"].values
    exog_future = split_data["X_future"].values

    # Auto-ARIMA с сезонностью m=4 (для квартальных данных)
    model = auto_arima(
        y_train,
        exogenous=exog_train,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        d=None,                # auto
        seasonal=True,
        m=4,                   # квартальная сезонность
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_fits=20
    )

    # Predict
    validation_pred = model.predict(n_periods=len(val_df), exogenous=exog_val)
    forecast_pred = model.predict(n_periods=len(future_df), exogenous=exog_future)

    return model, validation_pred, forecast_pred

def resolve_model_algorithm(model_request: Model) -> str:
    raw_value = (model_request.algorithm or model_request.name or "").strip().lower().replace("-", "_")
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
    # Adaptive CV folds based on dataset size
    if train_size < 10:
        cv_folds = 2
    elif train_size < 50:
        cv_folds = 3
    else:
        cv_folds = min(5, train_size // 5)

    if algorithm == "linear_regression":
        return LinearRegression()
    
    if algorithm == "lasso":
        # Data-adaptive alpha grid for Lasso
        alphas = np.logspace(-3, 2, 12) if train_size > 50 else np.logspace(-2, 1, 8)
        return LassoCV(alphas=alphas, cv=cv_folds, max_iter=50000, tol=1e-4, random_state=42)
    
    if algorithm == "ridge":
        # Data-adaptive alpha grid for Ridge
        alphas = np.logspace(-3, 2, 12) if train_size > 50 else np.logspace(-2, 1, 8)
        return RidgeCV(alphas=alphas, cv=cv_folds)
    
    if algorithm == "elasticnet":
        # Data-adaptive hyperparameters
        alphas = np.logspace(-3, 2, 12) if train_size > 50 else np.logspace(-2, 1, 8)
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
        return ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=cv_folds,
            max_iter=50000,
            tol=1e-4,
            random_state=42
        )
    
    if algorithm == "xgboost":
        # Aggressive regularization for small datasets
        if train_size < 50:
            n_estimators = 50  # Much smaller
            learning_rate = 0.2  # Higher to converge faster
            max_depth = 2  # Very shallow
            min_child_weight = 5  # Very restrictive
            subsample = 0.6  # More conservative sampling
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
            reg_alpha=0.1 if train_size < 50 else 0.01,  # Stronger L1 for small data
            reg_lambda=2.0 if train_size < 50 else 1.0,  # Stronger L2 for small data
            random_state=42,
            objective="reg:squarederror",
            early_stopping_rounds=5 if train_size > 50 else None
        )
    

    if algorithm == "prophet":
        return "prophet"
    if algorithm == "arima":
        return "arima"

    raise ValueError(f"Unsupported model algorithm '{algorithm}'")

def detect_optimal_lags(series: pd.Series, forecast_horizon: int = 1, max_lags: int = 12, significance_threshold: float = 1.96) -> list[int]:
    """
    Detect optimal lag periods using ACF (autocorrelation function).
    Returns lags that are statistically significant.
    """
    try:
        series_clean = series.dropna()
        if len(series_clean) < 10:
            return [1]
        
        # Calculate ACF up to max_lags
        acf_values = acf(series_clean, nlags=min(max_lags, len(series_clean) - 2), fft=False)
        
        # Confidence interval threshold (95% confidence)
        ci = significance_threshold / np.sqrt(len(series_clean))
        
        # Find significant lags (lags where ACF exceeds confidence interval)
        significant_lags = [i for i in range(1, len(acf_values)) 
                           if abs(acf_values[i]) > ci]
        
        if not significant_lags:
            return [1]  # Fallback to lag-1 if none significant
        
        # Ensure we have at least forecast_horizon lags
        min_lags = max(1, forecast_horizon)
        if max(significant_lags) < min_lags:
            significant_lags = list(range(1, min_lags + 1))
        
        return sorted(significant_lags[:4])  # Cap at 4 lags for performance
    except Exception as e:
        # Fallback to default lags if ACF fails
        return [1, 2]


def add_lag_features(
        frame: pd.DataFrame,
        target_col: str,
        lags: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    lag_cols = []
    for lag in lags:
        col_name = f"{target_col}_lag{lag}"
        frame[col_name] = frame[target_col].shift(lag)
        lag_cols.append(col_name)

    frame = frame.dropna(subset=lag_cols).reset_index(drop=True)
    return frame, lag_cols

def build_future_lag_features(
    frame: pd.DataFrame,
    target_col: str,
    lags: list[int],
    forecast_horizon: int,
) -> pd.DataFrame:
    target_history = list(frame[target_col].values)
    rows = []
    for h in range(forecast_horizon):
        row = {}
        for lag in lags:
            idx = len(target_history) - lag + h
            row[f"{target_col}_lag{lag}"] = target_history[max(0, min(idx, len(target_history) - 1))]
        rows.append(row)
    return pd.DataFrame(rows)

def normalize_forecast_frequency(value: str | None) -> str:
    normalized = (value or "Auto").strip().lower()
    if normalized == "quarterly":
        return "Quarterly"
    if normalized == "yearly":
        return "Yearly"
    if normalized == "auto":
        return "Auto"
    raise ValueError("Forecast frequency must be Auto, Yearly or Quarterly")


def detect_forecast_frequency(data: pd.DataFrame, value: str) -> str:
    if value != "Auto":
        return value

    if "Год" not in data.columns or "Квартал" not in data.columns:
        return "Yearly"

    quarter_values = pd.to_numeric(data["Квартал"], errors="coerce").dropna()
    if quarter_values.empty:
        return "Yearly"

    return "Quarterly" if quarter_values.between(1, 4).all() else "Yearly"


def prepare_yearly_training_data(
    data: pd.DataFrame,
    time_col: str,
    target_col: str,
    features: list[str],
    forecast_horizon: int,
) -> tuple[pd.DataFrame, np.ndarray, list[datetime], str]:
    if time_col not in data.columns:
        raise ValueError(f"Missing column '{time_col}'")

    frame = data[[time_col, target_col, *features]].copy()
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")

    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    frame = prepare_training_frame(frame, time_col, target_col, features)
    max_historical_period = int(frame[time_col].max())
    future_periods = np.arange(
        max_historical_period + 1,
        max_historical_period + forecast_horizon + 1,
        dtype=float,
    )
    forecast_timestamps = [
        datetime(int(period), 1, 1, tzinfo=timezone.utc)
        for period in future_periods
    ]
    return frame, future_periods, forecast_timestamps, time_col


def prepare_quarterly_training_data(
    data: pd.DataFrame,
    target_col: str,
    features: list[str],
    forecast_horizon: int,
) -> tuple[pd.DataFrame, np.ndarray, list[datetime]]:
    if "Год" not in data.columns or "Квартал" not in data.columns:
        raise ValueError("Quarterly forecast requires 'Год' and 'Квартал' columns")

    frame = data[["Год", "Квартал", target_col, *features]].copy()
    frame["Год"] = pd.to_numeric(frame["Год"], errors="coerce")
    frame["Квартал"] = pd.to_numeric(frame["Квартал"], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")

    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    frame = frame.dropna(subset=["Год", "Квартал", target_col]).copy()
    if frame.empty:
        raise ValueError("Dataset does not contain enough rows after removing empty time/target values")

    frame["Квартал"] = frame["Квартал"].astype(int)
    if not frame["Квартал"].between(1, 4).all():
        raise ValueError("Column 'Квартал' must contain values from 1 to 4")

    frame["__period_index"] = (frame["Год"].astype(int) * 4 + (frame["Квартал"] - 1)).astype(float)
    frame = frame.sort_values(["Год", "Квартал"]).reset_index(drop=True)

    for feature in features:
        if frame[feature].isna().all():
            raise ValueError(f"Column '{feature}' contains only empty values")

        frame[feature] = frame[feature].interpolate(method="linear", limit_direction="both")
        frame[feature] = frame[feature].ffill().bfill()

    frame = frame.dropna(subset=features).copy()
    if len(frame.index) < 2:
        raise ValueError("Dataset must contain at least two valid rows after filling empty values")

    last_year = int(frame.iloc[-1]["Год"])
    last_quarter = int(frame.iloc[-1]["Квартал"])

    future_periods = []
    forecast_timestamps = []
    current_year = last_year
    current_quarter = last_quarter
    current_index = int(frame.iloc[-1]["__period_index"])

    for _ in range(forecast_horizon):
        current_index += 1
        current_quarter += 1
        if current_quarter > 4:
            current_quarter = 1
            current_year += 1

        future_periods.append(float(current_index))
        forecast_timestamps.append(datetime(current_year, quarter_to_month(current_quarter), 1, tzinfo=timezone.utc))

    return frame, np.array(future_periods, dtype=float), forecast_timestamps


def quarter_to_month(quarter: int) -> int:
    return {1: 1, 2: 4, 3: 7, 4: 10}[quarter]


async def send_callback(callback_url: str, payload: TrainingRunCallbackRequest):
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        response = await client.post(callback_url, json=payload.model_dump(mode="json"))
        if response.is_success:
            return

        response_body = response.text.strip()
        if response_body:
            raise RuntimeError(
                f"Callback failed with status {response.status_code}: {response_body}"
            )

        raise RuntimeError(f"Callback failed with status {response.status_code}")
