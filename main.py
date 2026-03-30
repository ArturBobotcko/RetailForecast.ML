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
from models.Metric import Metric
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

app = FastAPI()


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

        if time_col not in data.columns or target_col not in data.columns:
            raise ValueError(f"Missing column '{time_col}' or '{target_col}'")

        if not pd.api.types.is_numeric_dtype(pd.to_numeric(data[time_col], errors="coerce")):
            raise ValueError(f"Column '{time_col}' must be numeric")

        if not pd.api.types.is_numeric_dtype(pd.to_numeric(data[target_col], errors="coerce")):
            raise ValueError(f"Column '{target_col}' must be numeric")

        features = [
            column for column in request.featureColumns
            if column in data.columns
            and column != time_col
            and column != target_col
            and pd.api.types.is_numeric_dtype(pd.to_numeric(data[column], errors="coerce"))
        ]

        if not features:
            features = [
                column for column in data.columns
                if column not in [time_col, target_col]
                and pd.api.types.is_numeric_dtype(pd.to_numeric(data[column], errors="coerce"))
            ]

        if not features:
            raise ValueError("No numeric feature columns found")

        frame = data[[time_col, target_col, *features]].copy()
        frame[time_col] = pd.to_numeric(frame[time_col], errors="raise")
        frame[target_col] = pd.to_numeric(frame[target_col], errors="raise")

        for feature in features:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")

        frame = frame.sort_values(time_col).reset_index(drop=True)

        max_historical_period = int(frame[time_col].max())
        forecast_period = request.forecastPeriod or (max_historical_period + 1)
        if forecast_period <= max_historical_period:
            raise ValueError(f"Forecast period must be greater than {max_historical_period}")

        future_periods = np.arange(max_historical_period + 1, forecast_period + 1)

        feature_models = {}
        for feature in features:
            lr = LinearRegression()
            lr.fit(frame[time_col].values.reshape(-1, 1), frame[feature].values)
            feature_models[feature] = lr

        future_x = pd.DataFrame({
            feature: feature_models[feature].predict(future_periods.reshape(-1, 1))
            for feature in features
        })

        scaler = StandardScaler()
        x_historical_scaled = scaler.fit_transform(frame[features])
        future_x_scaled = scaler.transform(future_x)

        lasso = Lasso(alpha=0.01, max_iter=10000)
        lasso.fit(x_historical_scaled, frame[target_col])

        historical_pred = lasso.predict(x_historical_scaled)
        forecast_pred = lasso.predict(future_x_scaled)

        actual = frame[target_col].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(actual - historical_pred)))
        rmse = float(np.sqrt(np.mean((actual - historical_pred) ** 2)))
        ss_res = float(np.sum((actual - historical_pred) ** 2))
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
                    "timestamp": datetime(int(period), 1, 1, tzinfo=timezone.utc).isoformat(),
                    "value": round(float(value), 4),
                }
                for period, value in zip(future_periods, forecast_pred, strict=True)
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
