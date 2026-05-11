# Executive Summary

This report was generated from a sample of `50,000` rows from `flights.csv`.

## Supervised Learning

- Best classification model: **Logistic Regression** with F1 `0.574` and ROC-AUC `0.751`.
- Best regression model: **Random Forest Regressor** with MAE `21.86` minutes and RMSE `40.47` minutes.

## Unsupervised Learning

Airport clusters were built from origin-airport profiles: traffic volume, delay rate, average delay, cancellation/diversion rates, route diversity and airline diversity.

|   cluster |   flights |   delay_rate |   avg_arrival_delay |   avg_departure_delay |   cancellation_rate |   diversion_rate |   avg_distance |   destinations |   airlines |
|----------:|----------:|-------------:|--------------------:|----------------------:|--------------------:|-----------------:|---------------:|---------------:|-----------:|
|         3 |    71.75  |        0.461 |              55.121 |                48.484 |               0.252 |            0     |        456.662 |          5.5   |      3     |
|         2 |   143.667 |        0.362 |              21.141 |                19.751 |               0.031 |            0.002 |        604.446 |         13.487 |      4.949 |
|         1 |  1208.6   |        0.287 |              12.47  |                15.138 |               0.013 |            0.002 |        985.537 |         80.28  |     10.56  |
|         0 |   217.286 |        0.221 |               7.508 |                 9.345 |               0.014 |            0.001 |        756.915 |         20.367 |      7.041 |

## Key Limitations

- The default run uses a sample to keep execution fast; final conclusions should be validated with the full dataset.
- The supervised models only use pre-flight variables to avoid leakage. This makes the task harder but more realistic.
- Weather, air-traffic and aircraft-rotation causes are only known after delays occur, so they are excluded from predictive features.
