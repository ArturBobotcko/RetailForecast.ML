from datetime import datetime, timezone

import pandas as pd
import pytest

from app.data.preparation import (
    detect_forecast_frequency,
    normalize_forecast_frequency,
    prepare_quarterly_training_data,
    prepare_yearly_training_data,
    select_feature_columns,
)


def test_normalize_forecast_frequency_accepts_supported_values():
    assert normalize_forecast_frequency(None) == "Auto"
    assert normalize_forecast_frequency(" quarterly ") == "Quarterly"
    assert normalize_forecast_frequency("YEARLY") == "Yearly"


def test_normalize_forecast_frequency_rejects_unknown_value():
    with pytest.raises(ValueError, match="Forecast frequency"):
        normalize_forecast_frequency("monthly")


def test_detect_forecast_frequency_uses_quarter_columns_for_auto():
    data = pd.DataFrame({"Год": [2024, 2024], "Квартал": [1, 2]})

    assert detect_forecast_frequency(data, "Auto") == "Quarterly"


def test_detect_forecast_frequency_falls_back_to_yearly():
    data = pd.DataFrame({"Год": [2024], "Квартал": [5]})

    assert detect_forecast_frequency(data, "Auto") == "Yearly"


def test_select_feature_columns_prefers_requested_numeric_features():
    data = pd.DataFrame(
        {
            "Год": [2023, 2024],
            "Квартал": [1, 2],
            "target": [10, 12],
            "requested": [1, 2],
            "text": ["a", "b"],
            "other": [3, 4],
        }
    )

    features = select_feature_columns(
        data=data,
        requested_features=["requested", "text"],
        time_col="Год",
        target_col="target",
        forecast_frequency="Auto",
    )

    assert features == ["requested"]


def test_prepare_yearly_training_data_fills_features_and_builds_future_periods():
    data = pd.DataFrame(
        {
            "Год": [2021, 2022, 2023],
            "target": [10, 12, 15],
            "feature": [1.0, None, 3.0],
        }
    )

    frame, future_periods, forecast_timestamps, time_col = prepare_yearly_training_data(
        data=data,
        time_col="Год",
        target_col="target",
        features=["feature"],
        forecast_horizon=2,
    )

    assert time_col == "Год"
    assert frame["feature"].isna().sum() == 0
    assert future_periods.tolist() == [2024.0, 2025.0]
    assert forecast_timestamps == [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    ]


def test_prepare_quarterly_training_data_builds_period_index_and_timestamps():
    data = pd.DataFrame(
        {
            "Год": [2024, 2024],
            "Квартал": [3, 4],
            "target": [10, 12],
            "feature": [1, 2],
        }
    )

    frame, future_periods, forecast_timestamps = prepare_quarterly_training_data(
        data=data,
        target_col="target",
        features=["feature"],
        forecast_horizon=2,
    )

    assert frame["__period_index"].tolist() == [8098.0, 8099.0]
    assert future_periods.tolist() == [8100.0, 8101.0]
    assert forecast_timestamps == [
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 4, 1, tzinfo=timezone.utc),
    ]
