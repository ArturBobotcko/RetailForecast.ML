import numpy as np
import pandas as pd
import pytest

from app.ml.features import add_lag_features, build_future_lag_features
from app.ml.metrics import calculate_validation_metrics
from app.ml.models import build_model, resolve_model_algorithm
from app.ml.regression_pipeline import run_regression_forecast
from app.ml.small_data_pipeline import (
    build_pipeline,
    evaluate_with_cv,
    split_features_target,
)
from models.Model import Model


def test_add_lag_features_drops_rows_without_history():
    frame = pd.DataFrame({"target": [10, 20, 30, 40]})

    result, lag_cols = add_lag_features(frame, "target", [1, 2])

    assert lag_cols == ["target_lag1", "target_lag2"]
    assert result["target"].tolist() == [30, 40]
    assert result["target_lag1"].tolist() == [20.0, 30.0]
    assert result["target_lag2"].tolist() == [10.0, 20.0]


def test_build_future_lag_features_uses_last_known_history_values():
    frame = pd.DataFrame(
        {
            "target": [10, 12, 15, 19, 24],
        }
    )

    result = build_future_lag_features(frame, "target", [1, 2], forecast_horizon=3)

    assert result["target_lag1"].tolist() == [24, 24, 24]
    assert result["target_lag2"].tolist() == [19, 24, 24]


def test_run_regression_forecast_returns_validation_and_final_forecast():
    frame = pd.DataFrame(
        {
            "Год": list(range(2012, 2024)),
            "target": [10, 12, 14, 17, 21, 25, 31, 38, 46, 55, 65, 76],
            "feature": [1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25],
        }
    )

    result = run_regression_forecast(
        model_request=Model(id=1, name="Linear", algorithm="linear_regression"),
        frame=frame,
        target_col="target",
        base_features=["feature"],
        time_col="Год",
        future_periods=np.array([2024.0, 2025.0]),
        forecast_horizon=2,
        validation_fraction=0.2,
    )

    assert len(result.validation_actual) >= 1
    assert len(result.validation_pred) == len(result.validation_actual)
    assert len(result.forecast_pred) == 2
    assert "target_lag1" in result.feature_columns


def test_regression_forecast_keeps_all_requested_features_from_origin_pipeline():
    row_count = 60
    frame = pd.DataFrame(
        {
            "Год": list(range(2000, 2000 + row_count)),
            "target": np.arange(row_count, dtype=float),
            "feature_a": np.arange(row_count, dtype=float),
            "feature_b": np.arange(row_count, dtype=float),
            "feature_c": np.arange(row_count, dtype=float),
            "feature_d": np.arange(row_count, dtype=float),
        }
    )

    result = run_regression_forecast(
        model_request=Model(id=1, name="Ridge", algorithm="ridge"),
        frame=frame,
        target_col="target",
        base_features=["feature_a", "feature_b", "feature_c", "feature_d"],
        time_col="Год",
        future_periods=np.array([2060.0]),
        forecast_horizon=1,
        validation_fraction=0.2,
    )

    assert result.feature_columns[:4] == [
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
    ]


def test_resolve_model_algorithm_supports_aliases():
    model = Model(id=1, name="Linear", algorithm="linear-regression")

    assert resolve_model_algorithm(model) == "linear_regression"


def test_lasso_and_elasticnet_aliases_use_regularized_models():
    lasso = Model(id=1, name="Lasso", algorithm="lasso")
    elasticnet = Model(id=2, name="ElasticNet", algorithm="elastic_net")

    assert resolve_model_algorithm(lasso) == "lasso"
    assert resolve_model_algorithm(elasticnet) == "elasticnet"


def test_complex_time_series_aliases_match_origin_pipeline():
    prophet = Model(id=1, name="Prophet", algorithm="prophet")
    arima = Model(id=2, name="ARIMA", algorithm="arima")

    assert resolve_model_algorithm(prophet) == "prophet"
    assert resolve_model_algorithm(arima) == "arima"


def test_xgboost_model_matches_origin_small_data_bounds():
    model = build_model("xgboost", train_size=60)

    assert model.n_estimators == 100
    assert model.learning_rate == 0.1
    assert model.max_depth == 3
    assert model.early_stopping_rounds == 5


def test_resolve_model_algorithm_rejects_unknown_algorithm():
    model = Model(id=1, name="Unknown", algorithm="unknown")

    with pytest.raises(ValueError, match="Unsupported model algorithm"):
        resolve_model_algorithm(model)


def test_calculate_validation_metrics_caps_mape_and_handles_zero_actuals():
    metrics = calculate_validation_metrics(
        actual=np.array([0.0, 100.0]),
        validation_pred=np.array([10.0, 50.0]),
    )

    by_name = {metric.name: metric.value for metric in metrics}
    assert by_name["mae"] == 30.0
    assert by_name["rmse"] == round(float(np.sqrt(1300.0)), 6)
    assert by_name["mape"] == 50.0


def test_small_data_pipeline_handles_mixed_features_with_cv():
    data = pd.DataFrame(
        {
            "store_type": [
                "A",
                "B",
                "A",
                "C",
                "B",
                "A",
                "C",
                "B",
                "A",
                "C",
                "B",
                "A",
            ],
            "promo": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
            "price_index": [
                1.0,
                0.9,
                1.1,
                0.8,
                0.95,
                1.2,
                1.15,
                0.85,
                1.05,
                0.75,
                1.0,
                0.9,
            ],
            "revenue": [
                100,
                140,
                108,
                170,
                150,
                105,
                130,
                168,
                112,
                185,
                122,
                155,
            ],
        }
    )

    x, y = split_features_target(data, target_column="revenue")
    pipeline = build_pipeline(x, model_name="random_forest")
    metrics = evaluate_with_cv(pipeline, x, y, n_splits=3, n_repeats=1)

    pipeline.fit(x, y)
    transformed = pipeline.named_steps["preprocess"].transform(x)

    assert metrics["fold_count"] == 3
    assert np.isfinite(metrics["mae_mean"])
    assert np.isfinite(metrics["rmse_mean"])
    assert np.isfinite(metrics["r2_mean"])
    assert transformed.shape[1] >= 3
