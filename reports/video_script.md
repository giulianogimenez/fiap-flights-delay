# Video Script

Suggested length: 5 to 10 minutes.

## 1. Opening

This project analyzes US flight delays and builds machine learning models to estimate whether a flight will arrive late and how long the delay may be.

Business question: can we anticipate delay risk before the flight using schedule, route, airline and airport information?

## 2. Dataset

The main dataset is `flights.csv`, with date, airline, origin and destination airports, scheduled times, actual times, cancellation/diversion flags and delay causes.

Important modeling decision: post-flight columns are not used as predictors. For example, `DEPARTURE_DELAY`, `ARRIVAL_TIME`, `AIR_TIME` and delay-cause columns would leak information that is unavailable before departure.

## 3. EDA

Show the generated figures:

- `reports/figures/arrival_delay_distribution.png`
- `reports/figures/delay_rate_by_hour.png`
- `reports/figures/delay_rate_by_airline.png`
- `reports/figures/top_origin_airports_delay_rate.png`

Explain which hours, airlines and origin airports concentrate higher delay rates, using the tables in `reports/tables/`.

## 4. Classification

Target: `IS_DELAYED = ARRIVAL_DELAY > 15`.

Models compared:

- Logistic Regression
- Random Forest

Current sample result:

- Logistic Regression: F1 around `0.574`, ROC-AUC around `0.751`.
- Random Forest: F1 around `0.570`, ROC-AUC around `0.750`.

Interpretation: both models perform similarly on the sample. Logistic Regression is easier to explain, while Random Forest can capture nonlinear patterns.

## 5. Regression

Target: `ARRIVAL_DELAY_MINUTES`.

Models compared:

- Ridge Regression
- Random Forest Regressor

Current sample result:

- Random Forest Regressor: MAE around `21.86` minutes.
- Ridge Regression: MAE around `22.98` minutes.

Interpretation: predicting exact minutes is harder than classifying delay risk, especially for extreme operational disruptions.

## 6. Unsupervised Learning

Airport profiles were grouped with K-Means using traffic volume, delay rate, average delays, cancellation/diversion rates, distance, destinations and airline diversity.

Show `reports/figures/airport_clusters_pca.png` and summarize `reports/tables/cluster_summary.csv`.

Interpretation: clusters help separate high-volume hubs, lower-delay airports and smaller airports with high delay severity.

## 7. Limitations

- The default run uses a sample to keep execution fast.
- Weather, aircraft rotation and air traffic causes are excluded from predictors because they are post-event columns in this dataset.
- The models should be validated on the full dataset and with stronger temporal validation before production use.

## 8. Next Steps

- Run the pipeline on all rows.
- Add external weather and holiday data.
- Tune boosting models.
- Build a dashboard for airport, route and airline monitoring.
