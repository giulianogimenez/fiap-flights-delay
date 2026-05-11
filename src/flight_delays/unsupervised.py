"""Unsupervised learning utilities for airport profiles."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flight_delays import config


PROFILE_FEATURES = [
    "flights",
    "delay_rate",
    "avg_arrival_delay",
    "avg_departure_delay",
    "cancellation_rate",
    "diversion_rate",
    "avg_distance",
    "destinations",
    "airlines",
]


def build_airport_profiles(df: pd.DataFrame, min_flights: int = 50) -> pd.DataFrame:
    """Aggregate flight behavior by origin airport."""

    profiled = df.copy()
    profiled["IS_DELAYED"] = profiled["ARRIVAL_DELAY"].gt(config.DELAY_THRESHOLD_MINUTES)
    profiles = (
        profiled.groupby("ORIGIN_AIRPORT")
        .agg(
            flights=("ORIGIN_AIRPORT", "size"),
            delay_rate=("IS_DELAYED", "mean"),
            avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
            avg_departure_delay=("DEPARTURE_DELAY", "mean"),
            cancellation_rate=("CANCELLED", "mean"),
            diversion_rate=("DIVERTED", "mean"),
            avg_distance=("DISTANCE", "mean"),
            destinations=("DESTINATION_AIRPORT", "nunique"),
            airlines=("AIRLINE", "nunique"),
        )
        .query("flights >= @min_flights")
        .reset_index()
    )
    return profiles


def cluster_airport_profiles(
    profiles: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, Pipeline]:
    """Cluster airport profiles and project them into two PCA components."""

    features = [feature for feature in PROFILE_FEATURES if feature in profiles.columns]
    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_scaled = preprocessing.fit_transform(profiles[features])
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    clusters = model.fit_predict(X_scaled)
    components = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

    clustered = profiles.copy()
    clustered["cluster"] = clusters
    clustered["pc1"] = components[:, 0]
    clustered["pc2"] = components[:, 1]
    pipeline = Pipeline(steps=[("preprocess", preprocessing), ("cluster", model)])
    return clustered, pipeline


def describe_clusters(clustered_profiles: pd.DataFrame) -> pd.DataFrame:
    """Summarize the behavioral meaning of each cluster."""

    summary_columns = [column for column in PROFILE_FEATURES if column in clustered_profiles.columns]
    return (
        clustered_profiles.groupby("cluster")[summary_columns]
        .mean()
        .round(3)
        .sort_values("delay_rate", ascending=False)
    )
