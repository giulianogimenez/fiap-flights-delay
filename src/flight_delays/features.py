"""Feature engineering for pre-flight delay prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flight_delays import config
from flight_delays.data import PREFLIGHT_FEATURE_COLUMNS, filter_completed_flights, normalize_flights


NUMERIC_FEATURES = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_DEPARTURE_MINUTES",
    "SCHEDULED_ARRIVAL_MINUTES",
    "DEPARTURE_HOUR",
    "ARRIVAL_HOUR",
    "MONTH_SIN",
    "MONTH_COS",
    "DOW_SIN",
    "DOW_COS",
]

CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "ROUTE",
    "DEPARTURE_PERIOD",
]


def hhmm_to_minutes(values: pd.Series) -> pd.Series:
    """Convert HHMM integer timestamps to minutes after midnight."""

    numeric = pd.to_numeric(values, errors="coerce")
    hours = np.floor(numeric / 100)
    minutes = numeric % 100
    valid = numeric.notna() & hours.between(0, 23) & minutes.between(0, 59)
    converted = hours * 60 + minutes
    return converted.where(valid)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and period-of-day variables available before departure."""

    featured = normalize_flights(df)
    featured["SCHEDULED_DEPARTURE_MINUTES"] = hhmm_to_minutes(featured["SCHEDULED_DEPARTURE"])
    featured["SCHEDULED_ARRIVAL_MINUTES"] = hhmm_to_minutes(featured["SCHEDULED_ARRIVAL"])
    featured["DEPARTURE_HOUR"] = (featured["SCHEDULED_DEPARTURE_MINUTES"] // 60).astype("Int64")
    featured["ARRIVAL_HOUR"] = (featured["SCHEDULED_ARRIVAL_MINUTES"] // 60).astype("Int64")

    featured["DEPARTURE_PERIOD"] = pd.cut(
        featured["DEPARTURE_HOUR"].astype(float),
        bins=[-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"],
    ).astype("object")

    featured["MONTH_SIN"] = np.sin(2 * np.pi * featured["MONTH"] / 12)
    featured["MONTH_COS"] = np.cos(2 * np.pi * featured["MONTH"] / 12)
    featured["DOW_SIN"] = np.sin(2 * np.pi * featured["DAY_OF_WEEK"] / 7)
    featured["DOW_COS"] = np.cos(2 * np.pi * featured["DAY_OF_WEEK"] / 7)
    return featured


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create classification and regression targets from arrival delay."""

    targeted = df.copy()
    targeted["IS_DELAYED"] = targeted["ARRIVAL_DELAY"].gt(config.DELAY_THRESHOLD_MINUTES).astype(int)
    targeted["ARRIVAL_DELAY_MINUTES"] = targeted["ARRIVAL_DELAY"]
    return targeted


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a modeling table using only pre-flight features plus targets."""

    completed = filter_completed_flights(df)
    featured = add_time_features(completed)
    targeted = create_targets(featured)
    available_columns = [
        column
        for column in PREFLIGHT_FEATURE_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if column in targeted.columns
    ]
    model_columns = list(dict.fromkeys(available_columns + ["IS_DELAYED", "ARRIVAL_DELAY_MINUTES", "FLIGHT_DATE"]))
    return targeted[model_columns].copy()


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical features present in a model frame."""

    numeric = [column for column in NUMERIC_FEATURES if column in df.columns]
    categorical = [column for column in CATEGORICAL_FEATURES if column in df.columns]
    return numeric, categorical
