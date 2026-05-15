import numpy as np
import pandas as pd


BASELINE_MODEL_CODES = {
    "last_value": 1,
    "seasonal_naive": 2,
}


def last_value_forecast(y_train: pd.Series, horizon: int) -> np.ndarray:
    if y_train.empty:
        raise ValueError("Cannot build LastValue baseline from empty training target")

    return np.full(horizon, float(y_train.iloc[-1]))


def seasonal_naive_forecast(
    y_train: pd.Series,
    horizon: int,
    season_length: int = 4,
) -> np.ndarray:
    if y_train.empty:
        raise ValueError("Cannot build SeasonalNaive baseline from empty training target")

    if len(y_train) < season_length:
        return last_value_forecast(y_train, horizon)

    pattern = y_train.iloc[-season_length:].to_numpy(dtype=float)
    repeats = int(np.ceil(horizon / season_length))

    return np.tile(pattern, repeats)[:horizon]


def build_baseline_forecast(
    y_train: pd.Series,
    horizon: int,
    resolved_frequency: str,
) -> tuple[str, np.ndarray]:
    if resolved_frequency == "Quarterly":
        return "seasonal_naive", seasonal_naive_forecast(
            y_train=y_train,
            horizon=horizon,
            season_length=4,
        )

    return "last_value", last_value_forecast(
        y_train=y_train,
        horizon=horizon,
    )
