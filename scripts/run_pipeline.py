"""Run the full Flight Delays analysis pipeline.

Example:
    python scripts/run_pipeline.py --nrows 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flight_delays import config
from flight_delays.data import basic_eda_tables, ensure_directories, load_flights, save_tables
from flight_delays.modeling import evaluate_classification, evaluate_regression
from flight_delays.reporting import (
    plot_clusters,
    plot_eda_figures,
    plot_model_metrics,
    write_markdown_summary,
)
from flight_delays.unsupervised import build_airport_profiles, cluster_airport_profiles, describe_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA, supervised and unsupervised analyses.")
    parser.add_argument("--flights-path", type=Path, default=config.FLIGHTS_PATH, help="Path to flights.csv.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=config.DEFAULT_SAMPLE_ROWS,
        help="Number of rows to read. Use 0 to read the complete dataset.",
    )
    parser.add_argument(
        "--min-airport-flights",
        type=int,
        default=50,
        help="Minimum flights per origin airport for cluster analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    nrows = None if args.nrows == 0 else args.nrows
    flights = load_flights(args.flights_path, nrows=nrows)
    sample_rows = len(flights)

    eda_tables = basic_eda_tables(flights)
    save_tables(eda_tables, config.REPORTS_DIR / "tables")
    plot_eda_figures(flights)

    classification_results = evaluate_classification(flights)
    regression_results = evaluate_regression(flights)
    classification_results.metrics.to_csv(config.REPORTS_DIR / "tables" / "classification_metrics.csv", index=False)
    regression_results.metrics.to_csv(config.REPORTS_DIR / "tables" / "regression_metrics.csv", index=False)
    plot_model_metrics(classification_results.metrics, regression_results.metrics)

    airport_profiles = build_airport_profiles(flights, min_flights=args.min_airport_flights)
    clustered_profiles, _ = cluster_airport_profiles(airport_profiles)
    cluster_summary = describe_clusters(clustered_profiles)
    airport_profiles.to_csv(config.REPORTS_DIR / "tables" / "airport_profiles.csv", index=False)
    clustered_profiles.to_csv(config.REPORTS_DIR / "tables" / "airport_clusters.csv", index=False)
    cluster_summary.to_csv(config.REPORTS_DIR / "tables" / "cluster_summary.csv")
    plot_clusters(clustered_profiles)

    write_markdown_summary(
        config.REPORTS_DIR / "executive_summary.md",
        sample_rows,
        classification_results.metrics,
        regression_results.metrics,
        cluster_summary,
    )

    print(f"Pipeline finished with {sample_rows:,} rows.")
    print(f"Reports written to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
