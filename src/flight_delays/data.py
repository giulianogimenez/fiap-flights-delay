"""Data loading, cleaning and descriptive analysis utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from flight_delays import config


POST_FLIGHT_COLUMNS = [
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "ARRIVAL_TIME",
    "ARRIVAL_DELAY",
    "ELAPSED_TIME",
    "AIR_TIME",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "CANCELLATION_REASON",
]

PREFLIGHT_FEATURE_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
]


def ensure_directories() -> None:
    """Create project output directories when running scripts."""

    for path in [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path: str | Path | None, fallback: Path) -> Path:
    """Resolve a user-provided or default path and fail with a helpful message."""

    candidate = Path(path).expanduser() if path else fallback
    if not candidate.exists():
        raise FileNotFoundError(
            f"Dataset not found at {candidate}. Pass a path with the CLI option "
            "or set FLIGHTS_CSV/AIRLINES_CSV/AIRPORTS_CSV."
        )
    return candidate


def load_flights(
    path: str | Path | None = None,
    nrows: int | None = config.DEFAULT_SAMPLE_ROWS,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the flights dataset.

    The original file is large, so scripts default to reading a representative
    prefix with `nrows`. Pass `nrows=None` to process the complete dataset.
    """

    csv_path = resolve_existing_path(path, config.FLIGHTS_PATH)
    return pd.read_csv(csv_path, nrows=nrows, usecols=usecols, low_memory=False)


def load_reference_data(
    airlines_path: str | Path | None = None,
    airports_path: str | Path | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load optional airline and airport lookup tables."""

    airlines = None
    airports = None
    if airlines_path or config.AIRLINES_PATH.exists():
        airlines = pd.read_csv(resolve_existing_path(airlines_path, config.AIRLINES_PATH))
    if airports_path or config.AIRPORTS_PATH.exists():
        airports = pd.read_csv(resolve_existing_path(airports_path, config.AIRPORTS_PATH))
    return airlines, airports


def normalize_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and add fields shared by EDA and modeling."""

    normalized = df.copy()
    normalized["ROUTE"] = (
        normalized["ORIGIN_AIRPORT"].astype(str)
        + "-"
        + normalized["DESTINATION_AIRPORT"].astype(str)
    )
    if {"YEAR", "MONTH", "DAY"}.issubset(normalized.columns):
        normalized["FLIGHT_DATE"] = pd.to_datetime(
            normalized[["YEAR", "MONTH", "DAY"]],
            errors="coerce",
        )
    return normalized


def filter_completed_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Keep flights where arrival delay is meaningful."""

    completed = df.copy()
    if "CANCELLED" in completed:
        completed = completed[completed["CANCELLED"].fillna(0).eq(0)]
    if "DIVERTED" in completed:
        completed = completed[completed["DIVERTED"].fillna(0).eq(0)]
    return completed[completed["ARRIVAL_DELAY"].notna()].copy()


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages per column."""

    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    return missing.sort_values("missing_pct", ascending=False)


def basic_eda_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build compact descriptive tables for the report."""

    completed = filter_completed_flights(df)
    delayed = completed.assign(IS_DELAYED=completed["ARRIVAL_DELAY"].gt(config.DELAY_THRESHOLD_MINUTES))
    by_airline = (
        delayed.groupby("AIRLINE")
        .agg(
            flights=("AIRLINE", "size"),
            delay_rate=("IS_DELAYED", "mean"),
            avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
            median_arrival_delay=("ARRIVAL_DELAY", "median"),
        )
        .sort_values(["delay_rate", "flights"], ascending=[False, False])
    )
    by_origin = (
        delayed.groupby("ORIGIN_AIRPORT")
        .agg(
            flights=("ORIGIN_AIRPORT", "size"),
            delay_rate=("IS_DELAYED", "mean"),
            avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        )
        .query("flights >= 30")
        .sort_values("delay_rate", ascending=False)
    )
    by_hour = (
        delayed.assign(SCHEDULED_HOUR=(delayed["SCHEDULED_DEPARTURE"].fillna(0) // 100).astype(int))
        .groupby("SCHEDULED_HOUR")
        .agg(
            flights=("SCHEDULED_HOUR", "size"),
            delay_rate=("IS_DELAYED", "mean"),
            avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        )
    )
    return {
        "missing_values": missing_value_report(df),
        "airline_delay_profile": by_airline,
        "origin_airport_delay_profile": by_origin,
        "scheduled_hour_delay_profile": by_hour,
    }


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist EDA/model tables as CSV files for reproducibility."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv")
