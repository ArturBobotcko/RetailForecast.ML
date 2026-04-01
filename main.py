import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from DTOs.TrainingRunRequest import TrainingRunRequest
from DTOs.TrainingRunCallbackRequest import TrainingRunCallbackRequest
from DTOs.TrainingRunResponse import TrainingRunResponse
from models.Model import Model
from models.Metric import Metric
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

app = FastAPI()
DEFAULT_VALIDATION_FRACTION = 0.2


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

        # Экстраполяция фичей
        feature_models = {}
        for feature in features:
            lr = LinearRegression()
            lr.fit(time_values, frame[feature].values)
            feature_models[feature] = lr

        future_x = pd.DataFrame({
            feature: feature_models[feature].predict(future_periods.reshape(-1, 1))
            for feature in features
        })

        split_data = split_and_scale_time_series_data(
            history_df=frame,
            future_df=future_x.assign(**{split_time_col: future_periods}),
            time_col=split_time_col,
            target_col=target_col,
            features=features,
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

        callback_request = TrainingRunCallbackRequest(
            status="Completed",
            metrics=[
                Metric(name="mae", value=round(mae, 6)),
                Metric(name="rmse", value=round(rmse, 6)),
                Metric(name="r2", value=round(r2, 6)),
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
            error=str(ex),
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

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[features]),
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
    use_scaled_features = algorithm in {"lasso", "ridge", "elasticnet", "linear_regression"}

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
    }

    if raw_value not in aliases:
        raise ValueError(
            f"Unsupported model algorithm '{model_request.algorithm or model_request.name}'. "
            "Supported values: linear_regression, lasso, ridge, elasticnet"
        )

    return aliases[raw_value]


def build_model(algorithm: str, train_size: int):
    cv_folds = min(3, max(2, train_size - 1))

    if algorithm == "linear_regression":
        return LinearRegression()
    if algorithm == "lasso":
        return LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=cv_folds, max_iter=10000)
    if algorithm == "ridge":
        return RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
    if algorithm == "elasticnet":
        return ElasticNetCV(
            alphas=[0.001, 0.01, 0.1, 1.0],
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=cv_folds,
            max_iter=10000,
        )

    raise ValueError(f"Unsupported model algorithm '{algorithm}'")


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
