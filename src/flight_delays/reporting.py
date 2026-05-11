"""Reporting utilities for figures and markdown summaries."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[2] / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from flight_delays import config
from flight_delays.data import filter_completed_flights


sns.set_theme(style="whitegrid")


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_eda_figures(df: pd.DataFrame, output_dir: Path = config.FIGURES_DIR) -> list[Path]:
    """Create core EDA figures required by the challenge."""

    completed = filter_completed_flights(df).copy()
    completed["IS_DELAYED"] = completed["ARRIVAL_DELAY"].gt(config.DELAY_THRESHOLD_MINUTES)
    completed["SCHEDULED_HOUR"] = (completed["SCHEDULED_DEPARTURE"].fillna(0) // 100).astype(int)
    paths: list[Path] = []

    plt.figure(figsize=(9, 5))
    sns.histplot(completed["ARRIVAL_DELAY"].clip(-60, 180), bins=60)
    plt.title("Distribution of Arrival Delay (clipped to -60..180 min)")
    plt.xlabel("Arrival delay in minutes")
    paths.append(output_dir / "arrival_delay_distribution.png")
    _save_current_figure(paths[-1])

    by_hour = completed.groupby("SCHEDULED_HOUR")["IS_DELAYED"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=by_hour, x="SCHEDULED_HOUR", y="IS_DELAYED", marker="o")
    plt.title("Delay Rate by Scheduled Departure Hour")
    plt.xlabel("Scheduled departure hour")
    plt.ylabel("Delay rate")
    paths.append(output_dir / "delay_rate_by_hour.png")
    _save_current_figure(paths[-1])

    by_airline = (
        completed.groupby("AIRLINE")
        .agg(delay_rate=("IS_DELAYED", "mean"), flights=("AIRLINE", "size"))
        .sort_values("delay_rate", ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=by_airline, x="AIRLINE", y="delay_rate")
    plt.title("Delay Rate by Airline")
    plt.xlabel("Airline")
    plt.ylabel("Delay rate")
    paths.append(output_dir / "delay_rate_by_airline.png")
    _save_current_figure(paths[-1])

    top_airports = (
        completed.groupby("ORIGIN_AIRPORT")
        .agg(delay_rate=("IS_DELAYED", "mean"), flights=("ORIGIN_AIRPORT", "size"))
        .query("flights >= 30")
        .sort_values("delay_rate", ascending=False)
        .head(15)
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_airports, y="ORIGIN_AIRPORT", x="delay_rate")
    plt.title("Top Origin Airports by Delay Rate")
    plt.xlabel("Delay rate")
    plt.ylabel("Origin airport")
    paths.append(output_dir / "top_origin_airports_delay_rate.png")
    _save_current_figure(paths[-1])
    return paths


def plot_model_metrics(
    classification_metrics: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    output_dir: Path = config.FIGURES_DIR,
) -> list[Path]:
    """Plot supervised model comparison charts."""

    paths: list[Path] = []
    plt.figure(figsize=(8, 5))
    sns.barplot(data=classification_metrics, x="model", y="f1")
    plt.title("Classification Model Comparison (F1)")
    plt.xlabel("")
    plt.ylabel("F1 score")
    plt.xticks(rotation=20, ha="right")
    paths.append(output_dir / "classification_f1_comparison.png")
    _save_current_figure(paths[-1])

    plt.figure(figsize=(8, 5))
    sns.barplot(data=regression_metrics, x="model", y="mae")
    plt.title("Regression Model Comparison (MAE)")
    plt.xlabel("")
    plt.ylabel("MAE in minutes")
    plt.xticks(rotation=20, ha="right")
    paths.append(output_dir / "regression_mae_comparison.png")
    _save_current_figure(paths[-1])
    return paths


def plot_clusters(clustered_profiles: pd.DataFrame, output_dir: Path = config.FIGURES_DIR) -> Path:
    """Plot clustered airport profiles in PCA space."""

    path = output_dir / "airport_clusters_pca.png"
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=clustered_profiles,
        x="pc1",
        y="pc2",
        hue="cluster",
        size="flights",
        sizes=(40, 220),
        palette="tab10",
    )
    plt.title("Airport Profiles: K-Means Clusters in PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    _save_current_figure(path)
    return path


def write_markdown_summary(
    output_path: Path,
    sample_rows: int,
    classification_metrics: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    """Write a concise executive summary with generated results."""

    best_classifier = classification_metrics.iloc[0]
    best_regressor = regression_metrics.iloc[0]
    content = f"""# Executive Summary

This report was generated from a sample of `{sample_rows:,}` rows from `flights.csv`.

## Supervised Learning

- Best classification model: **{best_classifier['model']}** with F1 `{best_classifier['f1']:.3f}` and ROC-AUC `{best_classifier['roc_auc']:.3f}`.
- Best regression model: **{best_regressor['model']}** with MAE `{best_regressor['mae']:.2f}` minutes and RMSE `{best_regressor['rmse']:.2f}` minutes.

## Unsupervised Learning

Airport clusters were built from origin-airport profiles: traffic volume, delay rate, average delay, cancellation/diversion rates, route diversity and airline diversity.

{cluster_summary.to_markdown()}

## Key Limitations

- The default run uses a sample to keep execution fast; final conclusions should be validated with the full dataset.
- The supervised models only use pre-flight variables to avoid leakage. This makes the task harder but more realistic.
- Weather, air-traffic and aircraft-rotation causes are only known after delays occur, so they are excluded from predictive features.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
