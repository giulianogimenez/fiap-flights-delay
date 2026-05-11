"""Supervised modeling helpers for classification and regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delays import config
from flight_delays.features import build_model_frame, feature_lists


@dataclass
class ModelResults:
    """Container for trained models and metric tables."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    models: dict[str, Pipeline]


def _one_hot_encoder() -> OneHotEncoder:
    """Create an encoder compatible with multiple scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical predictors."""

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def split_features_target(
    model_frame: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data, preferring a temporal holdout when dates are available."""

    numeric_features, categorical_features = feature_lists(model_frame)
    feature_columns = numeric_features + categorical_features
    modeling_data = model_frame.dropna(subset=[target]).copy()
    X = modeling_data[feature_columns]
    y = modeling_data[target]

    if "FLIGHT_DATE" in modeling_data and modeling_data["FLIGHT_DATE"].notna().nunique() > 10:
        cutoff = modeling_data["FLIGHT_DATE"].quantile(1 - test_size)
        train_mask = modeling_data["FLIGHT_DATE"].le(cutoff)
        if train_mask.mean() > 0.5 and (~train_mask).sum() > 100:
            return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]

    stratify = y if target == "IS_DELAYED" and y.nunique() == 2 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def classification_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Return the required pair of classification algorithms."""

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def regression_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Return the required pair of regression algorithms."""

    return {
        "Ridge Regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=10,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_classification(df: pd.DataFrame) -> ModelResults:
    """Train and evaluate two classifiers for delayed-flight prediction."""

    model_frame = build_model_frame(df)
    numeric_features, categorical_features = feature_lists(model_frame)
    X_train, X_test, y_train, y_test = split_features_target(model_frame, "IS_DELAYED")

    rows = []
    predictions = pd.DataFrame({"actual": y_test.reset_index(drop=True)})
    trained_models: dict[str, Pipeline] = {}
    for name, pipeline in classification_models(numeric_features, categorical_features).items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
            }
        )
        predictions[f"{name}_prediction"] = y_pred
        predictions[f"{name}_score"] = y_score
        trained_models[name] = pipeline
    return ModelResults(pd.DataFrame(rows).sort_values("f1", ascending=False), predictions, trained_models)


def evaluate_regression(df: pd.DataFrame) -> ModelResults:
    """Train and evaluate two regressors for arrival-delay minutes."""

    model_frame = build_model_frame(df)
    numeric_features, categorical_features = feature_lists(model_frame)
    X_train, X_test, y_train, y_test = split_features_target(model_frame, "ARRIVAL_DELAY_MINUTES")

    rows = []
    predictions = pd.DataFrame({"actual": y_test.reset_index(drop=True)})
    trained_models: dict[str, Pipeline] = {}
    for name, pipeline in regression_models(numeric_features, categorical_features).items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": r2_score(y_test, y_pred),
            }
        )
        predictions[f"{name}_prediction"] = y_pred
        trained_models[name] = pipeline
    return ModelResults(pd.DataFrame(rows).sort_values("mae"), predictions, trained_models)
