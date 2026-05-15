import traceback

from app.config import DEFAULT_VALIDATION_FRACTION
from app.data.loaders import load_data
from app.data.preparation import (
    detect_forecast_frequency,
    prepare_quarterly_training_data,
    prepare_yearly_training_data,
    select_feature_columns,
)
from app.infrastructure.callbacks import send_callback
from app.infrastructure.datasets import download_dataset
from app.ml.metrics import calculate_validation_metrics
from app.ml.regression_pipeline import run_regression_forecast
from DTOs.TrainingRunCallbackRequest import TrainingRunCallbackRequest


async def process_training_run(request, external_job_id: str) -> None:
    temp_file_path = None

    try:
        temp_file_path = await download_dataset(request.downloadUrl)
        data = load_data(temp_file_path)

        time_col = request.timeColumn.strip() if request.timeColumn else "Год"
        target_col = request.targetColumn.strip()

        features = select_feature_columns(
            data=data,
            requested_features=request.featureColumns,
            time_col=time_col,
            target_col=target_col,
            forecast_frequency=request.forecastFrequency,
        )

        forecast_horizon = request.forecastHorizon or 1
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be greater than 0")

        resolved_frequency = detect_forecast_frequency(data, request.forecastFrequency)

        if resolved_frequency == "Quarterly":
            frame, future_periods, forecast_timestamps = (
                prepare_quarterly_training_data(
                    data,
                    target_col,
                    features,
                    forecast_horizon,
                )
            )
            split_time_col = "__period_index"
        else:
            frame, future_periods, forecast_timestamps, resolved_time_col = (
                prepare_yearly_training_data(
                    data,
                    time_col,
                    target_col,
                    features,
                    forecast_horizon,
                )
            )
            split_time_col = resolved_time_col

        forecast_result = run_regression_forecast(
            model_request=request.model,
            frame=frame,
            target_col=target_col,
            base_features=features,
            time_col=split_time_col,
            future_periods=future_periods,
            forecast_horizon=forecast_horizon,
            validation_fraction=DEFAULT_VALIDATION_FRACTION,
        )

        callback_request = TrainingRunCallbackRequest(
            status="Completed",
            metrics=calculate_validation_metrics(
                forecast_result.validation_actual,
                forecast_result.validation_pred,
                forecast_result.training_target,
                resolved_frequency,
            ),
            forecast=[
                {
                    "timestamp": timestamp.isoformat(),
                    "value": round(float(value), 4),
                }
                for timestamp, value in zip(
                    forecast_timestamps, forecast_result.forecast_pred, strict=True
                )
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
