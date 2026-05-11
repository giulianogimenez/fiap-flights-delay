# Data

Place the challenge CSV files here if you want local project-relative paths:

- `data/raw/flights.csv`
- `data/raw/airlines.csv`
- `data/raw/airports.csv`

The scripts also auto-detect the same filenames in `~/Downloads`, which keeps large files out of Git. You can override paths with environment variables:

```bash
export FLIGHTS_CSV="/path/to/flights.csv"
export AIRLINES_CSV="/path/to/airlines.csv"
export AIRPORTS_CSV="/path/to/airports.csv"
```

Large raw and processed CSV files are ignored by `.gitignore`.
