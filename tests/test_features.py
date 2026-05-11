import pandas as pd

from flight_delays.features import build_model_frame, hhmm_to_minutes


def test_hhmm_to_minutes_handles_valid_and_invalid_values():
    values = pd.Series([0, 5, 1230, 2359, 2460, None])

    converted = hhmm_to_minutes(values)

    assert converted.iloc[0] == 0
    assert converted.iloc[1] == 5
    assert converted.iloc[2] == 750
    assert converted.iloc[3] == 1439
    assert pd.isna(converted.iloc[4])
    assert pd.isna(converted.iloc[5])


def test_build_model_frame_excludes_cancelled_and_diverted_flights():
    df = pd.DataFrame(
        {
            "YEAR": [2015, 2015, 2015],
            "MONTH": [1, 1, 1],
            "DAY": [1, 1, 1],
            "DAY_OF_WEEK": [4, 4, 4],
            "AIRLINE": ["AA", "AA", "DL"],
            "FLIGHT_NUMBER": [1, 2, 3],
            "TAIL_NUMBER": ["N1", "N2", "N3"],
            "ORIGIN_AIRPORT": ["JFK", "JFK", "ATL"],
            "DESTINATION_AIRPORT": ["LAX", "SFO", "MIA"],
            "SCHEDULED_DEPARTURE": [800, 900, 1000],
            "SCHEDULED_TIME": [300, 320, 120],
            "DISTANCE": [2475, 2586, 594],
            "SCHEDULED_ARRIVAL": [1100, 1230, 1200],
            "ARRIVAL_DELAY": [20, 10, 5],
            "CANCELLED": [0, 1, 0],
            "DIVERTED": [0, 0, 1],
        }
    )

    model_frame = build_model_frame(df)

    assert len(model_frame) == 1
    assert model_frame["IS_DELAYED"].iloc[0] == 1
    assert "DEPARTURE_DELAY" not in model_frame.columns
