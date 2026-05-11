"""Project configuration and dataset discovery helpers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_SAMPLE_ROWS = 100_000
RANDOM_STATE = 42
DELAY_THRESHOLD_MINUTES = 15


def default_download_path(filename: str) -> Path:
    """Return the expected file path in the user's Downloads folder."""

    return Path.home() / "Downloads" / filename


def configured_path(env_var: str, filename: str) -> Path:
    """Resolve a dataset path from an environment variable or Downloads."""

    return Path(os.getenv(env_var, default_download_path(filename))).expanduser()


FLIGHTS_PATH = configured_path("FLIGHTS_CSV", "flights.csv")
AIRLINES_PATH = configured_path("AIRLINES_CSV", "airlines.csv")
AIRPORTS_PATH = configured_path("AIRPORTS_CSV", "airports.csv")
