import pandas as pd
from src.mp267.cleaning import clean_data


def sample_data():
    return pd.DataFrame({
        "vehicle": ["Car", "THR", "Two"],
        "station_type": ["Highway", "Mall", "Residential"],
        "session_minutes": [120, 7200, 90],
        "energy_kwh": [25.0, 150.0, 30.0],
        "kwh_billed_in_app": [25.0, 150.0, 30.0],
    })


def test_vehicle_names_are_standardized():
    cleaned = clean_data(sample_data())
    assert cleaned["vehicle"].tolist() == [
        "car", "three_wheeler", "two_wheeler"
    ]


def test_mall_session_minutes_are_converted():
    cleaned = clean_data(sample_data())
    assert cleaned.loc[1, "session_minutes"] == 120


def test_leakage_removed_and_target_corrected():
    cleaned = clean_data(sample_data())
    assert "kwh_billed_in_app" not in cleaned.columns
    assert cleaned.loc[1, "energy_kwh"] == 15.0


def test_original_dataframe_is_not_modified():
    df = sample_data()
    original = df.copy(deep=True)
    clean_data(df)
    pd.testing.assert_frame_equal(df, original)
