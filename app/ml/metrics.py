import logging

import numpy as np
import pandas as pd

from app.ml.baselines import BASELINE_MODEL_CODES, build_baseline_forecast
from models.Metric import Metric

logger = logging.getLogger(__name__)


def calculate_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    mape_threshold = 1e-6
    mape_mask = np.abs(actual) > mape_threshold
    if mape_mask.sum() > 0:
        mape = float(
            np.mean(
                np.abs(
                    (actual[mape_mask] - predicted[mape_mask])
                    / actual[mape_mask]
                )
            )
            * 100
        )
        mape = min(mape, 1000.0)
    else:
        mape = 0.0

    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else float(1 - (ss_res / ss_tot))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def calculate_improvement_over_baseline(
    ml_mae: float,
    baseline_mae: float,
) -> float:
    if baseline_mae > 0:
        return ((baseline_mae - ml_mae) / baseline_mae) * 100

    return 0.0


def calculate_validation_metrics(
    actual,
    validation_pred,
    y_train: pd.Series | None = None,
    resolved_frequency: str | None = None,
) -> list[Metric]:
    actual = np.asarray(actual, dtype=float)
    validation_pred = np.asarray(validation_pred, dtype=float)
    ml_metrics = calculate_regression_metrics(actual, validation_pred)

    metrics = [
        Metric(name="mae", value=round(ml_metrics["mae"], 6)),
        Metric(name="rmse", value=round(ml_metrics["rmse"], 6)),
        Metric(name="r2", value=round(ml_metrics["r2"], 6)),
        Metric(name="mape", value=round(ml_metrics["mape"], 2)),
    ]

    if y_train is None or resolved_frequency is None:
        return metrics

    baseline_name, baseline_validation_pred = build_baseline_forecast(
        y_train=y_train,
        horizon=len(actual),
        resolved_frequency=resolved_frequency,
    )
    baseline_metrics = calculate_regression_metrics(
        actual=actual,
        predicted=baseline_validation_pred,
    )
    improvement_over_baseline = calculate_improvement_over_baseline(
        ml_mae=ml_metrics["mae"],
        baseline_mae=baseline_metrics["mae"],
    )

    logger.info(
        "Baseline comparison: model_mae=%s, baseline=%s, baseline_mae=%s, "
        "improvement=%.2f%%",
        ml_metrics["mae"],
        baseline_name,
        baseline_metrics["mae"],
        improvement_over_baseline,
    )

    baseline_model_code = BASELINE_MODEL_CODES[baseline_name]

    metrics.extend(
        [
            Metric(name="baseline_mae", value=round(baseline_metrics["mae"], 6)),
            Metric(name="baseline_rmse", value=round(baseline_metrics["rmse"], 6)),
            Metric(name="baseline_r2", value=round(baseline_metrics["r2"], 6)),
            Metric(name="baseline_mape", value=round(baseline_metrics["mape"], 2)),
            Metric(
                name="improvement_over_baseline_percent",
                value=round(improvement_over_baseline, 2),
            ),
            Metric(name="baseline_model_code", value=float(baseline_model_code)),
        ]
    )

    return metrics
