from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Dataset must be a .csv, .xlsx or .xls file")


def split_features_target(
    data: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in data.columns:
        raise ValueError(f"Missing target column '{target_column}'")

    y = pd.to_numeric(data[target_column], errors="coerce")
    valid_mask = y.notna()
    if valid_mask.sum() < 8:
        raise ValueError("At least eight rows with numeric target values are required")

    drop_columns = drop_columns or []
    columns_to_drop = [target_column, *[c for c in drop_columns if c in data.columns]]
    x = data.drop(columns=columns_to_drop)
    if x.empty:
        raise ValueError("No feature columns left after dropping target/ignored columns")

    return (
        x.loc[valid_mask].reset_index(drop=True),
        y.loc[valid_mask].astype(float).reset_index(drop=True),
    )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in x.columns if column not in numeric_features
    ]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=2,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns found")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_regressor(model_name: str = "random_forest"):
    model_name = model_name.strip().lower().replace("-", "_")

    if model_name in {"random_forest", "randomforest", "rf"}:
        return RandomForestRegressor(
            n_estimators=250,
            max_depth=4,
            min_samples_leaf=3,
            min_samples_split=6,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name in {"gradient_boosting", "gradientboosting", "gbr", "gbm"}:
        return GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )

    if model_name in {"extra_trees", "extratrees", "et"}:
        return ExtraTreesRegressor(
            n_estimators=250,
            max_depth=4,
            min_samples_leaf=3,
            min_samples_split=6,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name in {"knn", "k_neighbors", "kneighbors"}:
        return KNeighborsRegressor(n_neighbors=7, weights="distance")

    if model_name == "ridge":
        return Ridge(alpha=10.0)

    raise ValueError(
        "model_name must be one of: random_forest, gradient_boosting, "
        "extra_trees, knn, ridge"
    )


def build_pipeline(x: pd.DataFrame, model_name: str = "random_forest") -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(x)),
            ("model", build_regressor(model_name)),
        ]
    )


def evaluate_with_cv(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> dict[str, float | int]:
    n_splits = min(n_splits, len(x))
    if n_splits < 2:
        raise ValueError("At least two rows are required for cross-validation")

    if n_repeats <= 1:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
        )

    mae_scores = -cross_val_score(
        pipeline, x, y, cv=cv, scoring="neg_mean_absolute_error"
    )
    rmse_scores = -cross_val_score(
        pipeline, x, y, cv=cv, scoring="neg_root_mean_squared_error"
    )
    r2_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="r2")

    return {
        "fold_count": int(len(mae_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
    }


def train_evaluate_save(
    data_path: str | Path,
    target_column: str,
    output_path: str | Path,
    model_name: str = "random_forest",
    drop_columns: list[str] | None = None,
) -> dict[str, float | int | str]:
    data = load_dataset(data_path)
    x, y = split_features_target(data, target_column, drop_columns)
    pipeline = build_pipeline(x, model_name)
    cv_metrics = evaluate_with_cv(pipeline, x, y)

    pipeline.fit(x, y)
    encoded_feature_count = int(
        pipeline.named_steps["preprocess"].transform(x).shape[1]
    )

    artifact = {
        "pipeline": pipeline,
        "target_column": target_column,
        "feature_columns": list(x.columns),
        "model_name": model_name,
        "cv_metrics": cv_metrics,
        "encoded_feature_count": encoded_feature_count,
    }
    joblib.dump(artifact, output_path)

    return {
        **cv_metrics,
        "model_name": model_name,
        "rows": int(len(x)),
        "raw_feature_count": int(x.shape[1]),
        "encoded_feature_count": encoded_feature_count,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small-data retail regression pipeline."
    )
    parser.add_argument("--file", required=True, help="Path to .csv/.xlsx/.xls dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="retail_forecast_model.joblib")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "extra_trees", "knn", "ridge"],
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Column to ignore. Can be passed multiple times.",
    )
    args = parser.parse_args()

    metrics = train_evaluate_save(
        data_path=args.file,
        target_column=args.target,
        output_path=args.output,
        model_name=args.model,
        drop_columns=args.drop_column,
    )
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
