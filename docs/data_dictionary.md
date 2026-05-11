# Data Dictionary

The project uses the `flights.csv` dataset described in the challenge material.

| Column | Meaning | Type |
| --- | --- | --- |
| `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK` | Flight date fields | Integer |
| `AIRLINE` | Airline code | Categorical |
| `FLIGHT_NUMBER`, `TAIL_NUMBER` | Flight and aircraft identifiers | Integer/Text |
| `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT` | IATA airport codes | Categorical |
| `SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL` | Scheduled HHMM times | Integer |
| `DEPARTURE_TIME`, `ARRIVAL_TIME` | Actual HHMM times | Integer |
| `DEPARTURE_DELAY`, `ARRIVAL_DELAY` | Delay in minutes | Numeric |
| `TAXI_OUT`, `TAXI_IN`, `WHEELS_OFF`, `WHEELS_ON` | Operational timestamps and taxi durations | Numeric |
| `SCHEDULED_TIME`, `ELAPSED_TIME`, `AIR_TIME` | Flight duration fields | Numeric |
| `DISTANCE` | Distance in miles | Numeric |
| `DIVERTED`, `CANCELLED` | Diverted/cancelled flags | Binary |
| `CANCELLATION_REASON` | Cancellation reason code | Categorical |
| `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY` | Post-event delay causes | Numeric |

## Leakage Policy

The supervised models predict delay before the flight happens. Therefore, training features exclude actual departure/arrival times, operational taxi/wheel timestamps, realized delays and delay-cause columns. The target is derived from `ARRIVAL_DELAY`, but that column is not used as a predictor.
