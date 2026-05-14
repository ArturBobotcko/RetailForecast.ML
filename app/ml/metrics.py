import numpy as np

from models.Metric import Metric


def calculate_validation_metrics(actual, validation_pred) -> list[Metric]:
    mae = float(np.mean(np.abs(actual - validation_pred)))
    rmse = float(np.sqrt(np.mean((actual - validation_pred) ** 2)))
    ss_res = float(np.sum((actual - validation_pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else float(1 - (ss_res / ss_tot))

    print(f"  ss_res: {ss_res:.2f}, ss_tot: {ss_tot:.2f}, R²: {r2:.4f}")

    mape_threshold = 1e-6
    mape_mask = np.abs(actual) > mape_threshold
    if mape_mask.sum() > 0:
        mape = float(
            np.mean(
                np.abs(
                    (actual[mape_mask] - validation_pred[mape_mask])
                    / actual[mape_mask]
                )
            )
            * 100
        )
        mape = min(mape, 1000.0)
    else:
        mape = 0.0

    return [
        Metric(name="mae", value=round(mae, 6)),
        Metric(name="rmse", value=round(rmse, 6)),
        Metric(name="r2", value=round(r2, 6)),
        Metric(name="mape", value=round(mape, 2)),
    ]
