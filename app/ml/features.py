import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def detect_optimal_lags(
    series: pd.Series,
    forecast_horizon: int = 1,
    max_lags: int = 12,
    significance_threshold: float = 1.96,
) -> list[int]:
    try:
        series_clean = series.dropna()
        if len(series_clean) < 10:
            return [1]

        acf_values = acf(
            series_clean, nlags=min(max_lags, len(series_clean) - 2), fft=False
        )
        print(acf_values)
        ci = significance_threshold / np.sqrt(len(series_clean))

        significant_lags = [
            i for i in range(1, len(acf_values)) if abs(acf_values[i]) > ci
        ]

        if not significant_lags:
            return [1]

        min_lags = max(1, forecast_horizon)
        if max(significant_lags) < min_lags:
            significant_lags = list(range(1, min_lags + 1))

        return sorted(significant_lags[:4])
    except Exception:
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
    for horizon in range(forecast_horizon):
        row = {}
        for lag in lags:
            index = len(target_history) - lag + horizon
            row[f"{target_col}_lag{lag}"] = target_history[
                max(0, min(index, len(target_history) - 1))
            ]
        rows.append(row)

    return pd.DataFrame(rows)
