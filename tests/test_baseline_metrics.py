import numpy as np
import pandas as pd

from app.ml.baselines import (
    build_baseline_forecast,
    last_value_forecast,
    seasonal_naive_forecast,
)
from app.ml.metrics import (
    calculate_improvement_over_baseline,
    calculate_regression_metrics,
    calculate_validation_metrics,
)


def test_last_value_forecast_returns_last_training_value() -> None:
    forecast = last_value_forecast(pd.Series([10, 12, 15]), horizon=4)

    np.testing.assert_array_equal(forecast, np.array([15.0, 15.0, 15.0, 15.0]))


def test_seasonal_naive_forecast_repeats_last_four_values() -> None:
    forecast = seasonal_naive_forecast(
        pd.Series([1, 2, 3, 4, 5, 6]),
        horizon=6,
    )

    np.testing.assert_array_equal(forecast, np.array([3.0, 4.0, 5.0, 6.0, 3.0, 4.0]))


def test_seasonal_naive_forecast_falls_back_to_last_value() -> None:
    forecast = seasonal_naive_forecast(pd.Series([7, 8, 9]), horizon=3)

    np.testing.assert_array_equal(forecast, np.array([9.0, 9.0, 9.0]))


def test_calculate_regression_metrics() -> None:
    metrics = calculate_regression_metrics(
        actual=np.array([2.0, 4.0]),
        predicted=np.array([1.0, 5.0]),
    )

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["mape"] == 37.5
    assert metrics["r2"] == 0.0


def test_improvement_over_baseline_is_positive_when_ml_is_better() -> None:
    improvement = calculate_improvement_over_baseline(
        ml_mae=1.0,
        baseline_mae=10.0,
    )

    assert improvement == 90.0


def test_validation_metrics_include_yearly_baseline_code() -> None:
    metrics = calculate_validation_metrics(
        actual=np.array([10.0, 20.0]),
        validation_pred=np.array([11.0, 19.0]),
        y_train=pd.Series([5.0, 5.0, 5.0, 5.0]),
        resolved_frequency="Yearly",
    )
    metric_values = {metric.name: metric.value for metric in metrics}

    assert metric_values["baseline_model_code"] == 1.0
    assert metric_values["improvement_over_baseline_percent"] > 0


def test_quarterly_frequency_uses_seasonal_naive_baseline() -> None:
    baseline_name, forecast = build_baseline_forecast(
        y_train=pd.Series([1, 2, 3, 4, 5, 6]),
        horizon=2,
        resolved_frequency="Quarterly",
    )

    assert baseline_name == "seasonal_naive"
    np.testing.assert_array_equal(forecast, np.array([3.0, 4.0]))
