from datetime import datetime, timezone

import numpy as np
import pandas as pd


def select_feature_columns(
    data: pd.DataFrame,
    requested_features: list[str],
    time_col: str,
    target_col: str,
    forecast_frequency: str | None,
) -> list[str]:
    forecast_frequency = normalize_forecast_frequency(forecast_frequency)

    if target_col not in data.columns:
        raise ValueError(f"Missing column '{target_col}'")

    if not _has_numeric_values(data[target_col]):
        raise ValueError(f"Column '{target_col}' must be numeric")

    reserved_time_columns = {time_col, target_col}
    if forecast_frequency in {"Auto", "Quarterly"}:
        reserved_time_columns.update({"Год", "Квартал"})

    features = [
        column
        for column in requested_features
        if column in data.columns
        and column not in reserved_time_columns
        and _has_numeric_values(data[column])
    ]

    if not features:
        features = [
            column
            for column in data.columns
            if column not in reserved_time_columns
            and _has_numeric_values(data[column])
        ]

    if not features:
        raise ValueError("No numeric feature columns found")

    return features


def _has_numeric_values(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().any()


def prepare_training_frame(
    frame: pd.DataFrame, time_col: str, target_col: str, features: list[str]
) -> pd.DataFrame:
    frame = frame.dropna(subset=[time_col, target_col]).copy()
    if frame.empty:
        raise ValueError(
            "Dataset does not contain enough rows after removing empty time/target values"
        )

    frame = frame.sort_values(time_col).reset_index(drop=True)

    for feature in features:
        if frame[feature].isna().all():
            raise ValueError(f"Column '{feature}' contains only empty values")

        frame[feature] = frame[feature].interpolate(
            method="linear", limit_direction="both"
        )
        frame[feature] = frame[feature].ffill().bfill()

    frame = frame.dropna(subset=features).copy()
    if len(frame.index) < 2:
        raise ValueError(
            "Dataset must contain at least two valid rows after filling empty values"
        )

    return frame


def normalize_forecast_frequency(value: str | None) -> str:
    normalized = (value or "Auto").strip().lower()
    if normalized == "quarterly":
        return "Quarterly"
    if normalized == "yearly":
        return "Yearly"
    if normalized == "auto":
        return "Auto"
    raise ValueError("Forecast frequency must be Auto, Yearly or Quarterly")


def detect_forecast_frequency(data: pd.DataFrame, value: str | None) -> str:
    value = normalize_forecast_frequency(value)
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
        datetime(int(period), 1, 1, tzinfo=timezone.utc) for period in future_periods
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
        raise ValueError(
            "Dataset does not contain enough rows after removing empty time/target values"
        )

    frame["Квартал"] = frame["Квартал"].astype(int)
    if not frame["Квартал"].between(1, 4).all():
        raise ValueError("Column 'Квартал' must contain values from 1 to 4")

    frame["__period_index"] = (
        frame["Год"].astype(int) * 4 + (frame["Квартал"] - 1)
    ).astype(float)
    frame = frame.sort_values(["Год", "Квартал"]).reset_index(drop=True)

    for feature in features:
        if frame[feature].isna().all():
            raise ValueError(f"Column '{feature}' contains only empty values")

        frame[feature] = frame[feature].interpolate(
            method="linear", limit_direction="both"
        )
        frame[feature] = frame[feature].ffill().bfill()

    frame = frame.dropna(subset=features).copy()
    if len(frame.index) < 2:
        raise ValueError(
            "Dataset must contain at least two valid rows after filling empty values"
        )

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
        forecast_timestamps.append(
            datetime(
                current_year, quarter_to_month(current_quarter), 1, tzinfo=timezone.utc
            )
        )

    return frame, np.array(future_periods, dtype=float), forecast_timestamps


def quarter_to_month(quarter: int) -> int:
    return {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
