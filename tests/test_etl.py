"""Tests for ETL module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.etl.transformer import (clean_lap_data, load_all_lap_data,
                              load_track_status_data, load_weather_data,
                              merge_track_status, merge_weather, split_train_test)


@pytest.fixture
def mock_lap_data():
    """Create mock lap data."""
    return pd.DataFrame({
        "RaceId": ["2023_01_Bahrain_R", "2023_01_Bahrain_R", "2023_01_Bahrain_R"],
        "LapNumber": [1, 2, 3],
        "Driver": ["HAM", "HAM", "HAM"],
        "Team": ["Mercedes", "Mercedes", "Mercedes"],
        "LapTime": [pd.Timedelta(seconds=90), pd.Timedelta(seconds=89), pd.Timedelta(seconds=91)],
        "Compound": ["SOFT", "SOFT", "SOFT"],
        "TyreLife": [1, 2, 3],
        "Position": [3, 2, 2],
        "IsAccurate": [True, True, True],
        "LapStartTime": [
            pd.Timestamp("2023-03-05 15:10:00"),
            pd.Timestamp("2023-03-05 15:11:30"),
            pd.Timestamp("2023-03-05 15:13:00"),
        ],
        "SessionType": ["R", "R", "R"],
        "EventName": ["Bahrain_Grand_Prix"],
    })


@pytest.fixture
def mock_track_status_data():
    """Create mock track status data."""
    return pd.DataFrame({
        "RaceId": ["2023_01_Bahrain_R", "2023_01_Bahrain_R"],
        "Time": [
            pd.Timestamp("2023-03-05 15:12:00"),  # During lap 2
            pd.Timestamp("2023-03-05 15:14:00"),  # During lap 3
        ],
        "Status": ["2", "4"],  # Yellow flag, Safety car
    })


@pytest.fixture
def mock_weather_data():
    """Create mock weather data."""
    return pd.DataFrame({
        "RaceId": ["2023_01_Bahrain_R", "2023_01_Bahrain_R"],
        "Time": [
            pd.Timestamp("2023-03-05 15:09:00"),  # Before lap 1
            pd.Timestamp("2023-03-05 15:12:30"),  # During lap 2
        ],
        "AirTemp": [28.5, 29.0],
        "Humidity": [45.0, 44.0],
        "Rainfall": [0.0, 0.0],
        "TrackTemp": [36.0, 37.0],
    })


def test_load_all_lap_data(mock_lap_data):
    """Test loading lap data from files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock lap files
        for race in ["2023_01_Bahrain", "2023_02_Jeddah"]:
            df = mock_lap_data.copy()
            df["RaceId"] = f"{race}_R"
            df.to_parquet(f"{tmpdir}/{race}_R_laps.parquet")
        
        # Patch RAW_DATA_DIR to use temp directory
        with patch("src.etl.transformer.RAW_DATA_DIR", Path(tmpdir)):
            # Test loading lap data
            result = load_all_lap_data()
            
            # Verify results
            assert len(result) == 6  # 3 laps x 2 races
            assert "RaceId" in result.columns
            assert set(result["RaceId"].unique()) == {"2023_01_Bahrain_R", "2023_02_Jeddah_R"}


def test_merge_track_status(mock_lap_data, mock_track_status_data):
    """Test merging track status data."""
    result = merge_track_status(mock_lap_data, mock_track_status_data)
    
    # Verify results
    assert "SafetyCar" in result.columns
    assert "YellowFlag" in result.columns
    assert "VirtualSafetyCar" in result.columns
    assert "RedFlag" in result.columns
    
    # Check that the yellow flag was set on lap 2
    lap2_row = result[result["LapNumber"] == 2]
    assert lap2_row["YellowFlag"].iloc[0] == True
    assert lap2_row["SafetyCar"].iloc[0] == False
    
    # Check that the safety car was set on lap 3
    lap3_row = result[result["LapNumber"] == 3]
    assert lap3_row["SafetyCar"].iloc[0] == True


def test_merge_weather(mock_lap_data, mock_weather_data):
    """Test merging weather data."""
    result = merge_weather(mock_lap_data, mock_weather_data)
    
    # Verify results
    assert "AirTemp" in result.columns
    assert "Humidity" in result.columns
    assert "TrackTemp" in result.columns
    
    # Check that lap 1 got weather from the first measurement
    lap1_row = result[result["LapNumber"] == 1]
    assert lap1_row["AirTemp"].iloc[0] == 28.5
    
    # Check that lap 2 got weather from the second measurement
    lap2_row = result[result["LapNumber"] == 2]
    assert lap2_row["AirTemp"].iloc[0] == 29.0
    
    # Check that lap 3 also got weather from the second measurement (last available)
    lap3_row = result[result["LapNumber"] == 3]
    assert lap3_row["AirTemp"].iloc[0] == 29.0


def test_clean_lap_data(mock_lap_data):
    """Test cleaning lap data."""
    # Add some missing values and an invalid lap
    test_data = mock_lap_data.copy()
    test_data.loc[1, "TyreLife"] = None
    test_data.loc[2, "TrackId"] = "Bahrain"
    
    # Add a row with invalid lap (not accurate)
    invalid_lap = test_data.iloc[0].copy()
    invalid_lap["LapNumber"] = 4
    invalid_lap["IsAccurate"] = False
    test_data = pd.concat([test_data, pd.DataFrame([invalid_lap])], ignore_index=True)
    
    result = clean_lap_data(test_data)
    
    # Verify results
    assert "TrackLength" in result.columns
    assert "TrackType" in result.columns
    assert "Corners" in result.columns
    assert "LapTime_seconds" in result.columns
    
    # Check that invalid lap was removed
    assert 4 not in result["LapNumber"].values
    
    # Check that missing values were filled
    assert not result["TyreLife"].isna().any()
    
    # Check that track info was added
    assert not result["TrackLength"].isna().any()
    assert not result["Corners"].isna().any()


def test_split_train_test(mock_lap_data):
    """Test splitting data into train, test, and holdout sets."""
    # Prepare test data with different years
    test_data = pd.concat([
        # 2022 data
        pd.DataFrame({
            "RaceId": ["2022_01_Bahrain_R", "2022_01_Bahrain_R"],
            "LapNumber": [1, 2],
            "Year": [2022, 2022],
            "sc_next_lap": [0, 0],
        }),
        # 2023 data
        pd.DataFrame({
            "RaceId": ["2023_01_Bahrain_R", "2023_01_Bahrain_R"],
            "LapNumber": [1, 2],
            "Year": [2023, 2023],
            "sc_next_lap": [0, 1],
        }),
        # 2024 data (holdout)
        pd.DataFrame({
            "RaceId": ["2024_01_Bahrain_R", "2024_01_Bahrain_R"],
            "LapNumber": [1, 2],
            "Year": [2024, 2024],
            "sc_next_lap": [1, 0],
        }),
    ])
    
    # Mock holdout years config
    with patch("src.etl.transformer.HOLDOUT_YEARS", {2024}):
        # Split the data
        train_df, test_df, holdout_df = split_train_test(test_data)
        
        # Verify results
        assert len(train_df) + len(test_df) + len(holdout_df) == len(test_data)
        
        # Check that holdout only has 2024 data
        assert holdout_df["Year"].unique() == [2024]
        
        # Check that train and test only have pre-2024 data
        assert set(train_df["Year"].unique()) | set(test_df["Year"].unique()) == {2022, 2023} 