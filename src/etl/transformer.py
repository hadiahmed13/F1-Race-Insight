"""Data transformer for F1 Race Insight."""

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (PROCESSED_DATASET_PATH, RAW_DATA_DIR, TARGET_COLUMN,
                       HOLDOUT_DATASET_PATH, TRAIN_DATASET_PATH, TEST_DATASET_PATH,
                       TEST_SIZE, HOLDOUT_YEARS)
from src.features.target import add_safety_car_target
from src.utils.io import load_dataframe, save_dataframe
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_all_lap_data() -> pd.DataFrame:
    """Load all lap data from raw files.

    Returns:
        DataFrame with all lap data.
    """
    logger.info("Loading all lap data files")
    lap_files = list(RAW_DATA_DIR.glob("*_laps.parquet"))
    
    if not lap_files:
        raise FileNotFoundError("No lap data files found in raw data directory")
    
    # Load and concatenate all lap data files
    dfs = []
    for file in lap_files:
        try:
            df = load_dataframe(file)
            race_id = file.stem.replace("_laps", "")
            df["RaceId"] = race_id
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading lap data file {file}", error=str(e))
    
    # Concatenate all dataframes
    all_laps_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(dfs)} lap data files", total_laps=len(all_laps_df))
    
    return all_laps_df


def load_track_status_data() -> pd.DataFrame:
    """Load all track status data from raw files.

    Returns:
        DataFrame with all track status data.
    """
    logger.info("Loading all track status files")
    track_status_files = list(RAW_DATA_DIR.glob("*_track_status.parquet"))
    
    if not track_status_files:
        logger.warning("No track status files found in raw data directory")
        return pd.DataFrame()
    
    # Load and concatenate all track status files
    dfs = []
    for file in track_status_files:
        try:
            df = load_dataframe(file)
            race_id = file.stem.replace("_track_status", "")
            df["RaceId"] = race_id
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading track status file {file}", error=str(e))
    
    # Concatenate all dataframes if any were loaded
    if dfs:
        all_track_status_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(dfs)} track status files", total_records=len(all_track_status_df))
        return all_track_status_df
    else:
        return pd.DataFrame()


def load_weather_data() -> pd.DataFrame:
    """Load all weather data from raw files.

    Returns:
        DataFrame with all weather data.
    """
    logger.info("Loading all weather data files")
    weather_files = list(RAW_DATA_DIR.glob("*_weather.parquet"))
    
    if not weather_files:
        logger.warning("No weather files found in raw data directory")
        return pd.DataFrame()
    
    # Load and concatenate all weather files
    dfs = []
    for file in weather_files:
        try:
            df = load_dataframe(file)
            race_id = file.stem.replace("_weather", "")
            df["RaceId"] = race_id
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading weather file {file}", error=str(e))
    
    # Concatenate all dataframes if any were loaded
    if dfs:
        all_weather_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(dfs)} weather files", total_records=len(all_weather_df))
        return all_weather_df
    else:
        return pd.DataFrame()


def merge_track_status(laps_df: pd.DataFrame, track_status_df: pd.DataFrame) -> pd.DataFrame:
    """Merge track status information into lap data.

    Args:
        laps_df: DataFrame with lap data.
        track_status_df: DataFrame with track status data.

    Returns:
        DataFrame with merged data.
    """
    if track_status_df.empty:
        logger.warning("No track status data to merge")
        
        # Add empty safety car columns
        for col in ["SafetyCar", "VirtualSafetyCar", "YellowFlag", "RedFlag"]:
            laps_df[col] = False
        
        return laps_df
    
    logger.info("Merging track status data")
    
    # Make a copy to avoid modifying the original
    result_df = laps_df.copy()
    
    # Add safety car and other flag columns
    result_df["SafetyCar"] = False
    result_df["VirtualSafetyCar"] = False
    result_df["YellowFlag"] = False
    result_df["RedFlag"] = False
    
    # Process each unique race
    for race_id in result_df["RaceId"].unique():
        race_track_status = track_status_df[track_status_df["RaceId"] == race_id]
        
        if race_track_status.empty:
            continue
        
        # Get lap data for this race
        race_laps = result_df[result_df["RaceId"] == race_id]
        
        # Process each lap
        for _, lap in race_laps.iterrows():
            lap_start = lap["LapStartTime"]
            
            if pd.isnull(lap_start):
                continue
            
            # Get end time (either next lap start or a fixed duration after lap start)
            next_lap = race_laps[
                (race_laps["Driver"] == lap["Driver"]) &
                (race_laps["LapNumber"] == lap["LapNumber"] + 1)
            ]
            
            if not next_lap.empty and not pd.isnull(next_lap["LapStartTime"].iloc[0]):
                lap_end = next_lap["LapStartTime"].iloc[0]
            elif not pd.isnull(lap["LapTime"]):
                # If no next lap, use lap time
                lap_end = lap_start + pd.Timedelta(seconds=lap["LapTime"].total_seconds())
            else:
                # If no lap time, use a fixed duration (e.g., 2 minutes)
                lap_end = lap_start + pd.Timedelta(minutes=2)
            
            # Find track status changes during this lap
            lap_status_changes = race_track_status[
                (race_track_status["Time"] >= lap_start) &
                (race_track_status["Time"] < lap_end)
            ]
            
            if not lap_status_changes.empty:
                # Check for different track statuses
                for _, status in lap_status_changes.iterrows():
                    if status["Status"] == "1":  # Track clear
                        pass
                    elif status["Status"] == "2":  # Yellow flag
                        result_df.loc[
                            (result_df["RaceId"] == race_id) &
                            (result_df["Driver"] == lap["Driver"]) &
                            (result_df["LapNumber"] == lap["LapNumber"]),
                            "YellowFlag"
                        ] = True
                    elif status["Status"] == "4":  # Safety car
                        result_df.loc[
                            (result_df["RaceId"] == race_id) &
                            (result_df["Driver"] == lap["Driver"]) &
                            (result_df["LapNumber"] == lap["LapNumber"]),
                            "SafetyCar"
                        ] = True
                    elif status["Status"] == "5":  # Red flag
                        result_df.loc[
                            (result_df["RaceId"] == race_id) &
                            (result_df["Driver"] == lap["Driver"]) &
                            (result_df["LapNumber"] == lap["LapNumber"]),
                            "RedFlag"
                        ] = True
                    elif status["Status"] == "6":  # Virtual safety car
                        result_df.loc[
                            (result_df["RaceId"] == race_id) &
                            (result_df["Driver"] == lap["Driver"]) &
                            (result_df["LapNumber"] == lap["LapNumber"]),
                            "VirtualSafetyCar"
                        ] = True
    
    logger.info("Track status data merged")
    return result_df


def merge_weather(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather information into lap data.

    Args:
        laps_df: DataFrame with lap data.
        weather_df: DataFrame with weather data.

    Returns:
        DataFrame with merged data.
    """
    if weather_df.empty:
        logger.warning("No weather data to merge")
        return laps_df
    
    logger.info("Merging weather data")
    
    # Make a copy to avoid modifying the original
    result_df = laps_df.copy()
    
    # Process each unique race
    for race_id in result_df["RaceId"].unique():
        race_weather = weather_df[weather_df["RaceId"] == race_id]
        
        if race_weather.empty:
            continue
        
        # Get lap data for this race
        race_laps = result_df[result_df["RaceId"] == race_id]
        
        # Process each lap
        for _, lap in race_laps.iterrows():
            lap_start = lap["LapStartTime"]
            
            if pd.isnull(lap_start):
                continue
            
            # Find closest weather measurement before lap start
            closest_weather = race_weather[race_weather["Time"] <= lap_start].sort_values("Time").iloc[-1:]
            
            if not closest_weather.empty:
                # Columns to merge
                weather_cols = ["AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindDirection", "WindSpeed"]
                
                for col in weather_cols:
                    if col in closest_weather.columns:
                        result_df.loc[
                            (result_df["RaceId"] == race_id) &
                            (result_df["Driver"] == lap["Driver"]) &
                            (result_df["LapNumber"] == lap["LapNumber"]),
                            col
                        ] = closest_weather[col].iloc[0]
    
    # Fill missing weather values with session medians
    for col in ["AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindSpeed"]:
        if col in result_df.columns and result_df[col].isna().any():
            # Group by race and compute median
            medians = result_df.groupby("RaceId")[col].median()
            
            # Fill NaNs with the median for their race
            for race_id in result_df["RaceId"].unique():
                if race_id in medians and not pd.isna(medians[race_id]):
                    mask = (result_df["RaceId"] == race_id) & (result_df[col].isna())
                    result_df.loc[mask, col] = medians[race_id]
    
    logger.info("Weather data merged")
    return result_df


def clean_lap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean lap data by handling missing values and outliers.

    Args:
        df: DataFrame with lap data.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning lap data")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Filter out rows with NaN lap times for race sessions (keep for qualifying)
    result_df = result_df[
        (result_df["SessionType"] == "Q") |
        ((result_df["SessionType"] == "R") & (~pd.isna(result_df["LapTime"])))
    ]
    
    # Filter out invalid laps (e.g., in-lap, out-lap) for race sessions
    result_df = result_df[
        (result_df["SessionType"] == "Q") |
        ((result_df["SessionType"] == "R") & (result_df["IsAccurate"]))
    ]
    
    # Handle missing values
    # Convert lap time columns from timedelta to seconds
    for time_col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "PitDuration"]:
        if time_col in result_df.columns:
            # Convert timedelta to seconds
            result_df[f"{time_col}_seconds"] = result_df[time_col].apply(
                lambda x: x.total_seconds() if not pd.isna(x) else np.nan
            )
    
    # Add track length and corners for each circuit
    track_info = {
        "Albert_Park": {"TrackLength": 5.278, "TrackType": "Street", "Corners": 14},
        "Bahrain": {"TrackLength": 5.412, "TrackType": "Permanent", "Corners": 15},
        "Miami": {"TrackLength": 5.412, "TrackType": "Street", "Corners": 19},
        "Imola": {"TrackLength": 4.909, "TrackType": "Permanent", "Corners": 19},
        "Monaco": {"TrackLength": 3.337, "TrackType": "Street", "Corners": 19},
        "Barcelona": {"TrackLength": 4.675, "TrackType": "Permanent", "Corners": 16},
        "Montréal": {"TrackLength": 4.361, "TrackType": "Street", "Corners": 14},
        "Silverstone": {"TrackLength": 5.891, "TrackType": "Permanent", "Corners": 18},
        "Red_Bull_Ring": {"TrackLength": 4.318, "TrackType": "Permanent", "Corners": 10},
        "Hungaroring": {"TrackLength": 4.381, "TrackType": "Permanent", "Corners": 14},
        "Spa-Francorchamps": {"TrackLength": 7.004, "TrackType": "Permanent", "Corners": 19},
        "Zandvoort": {"TrackLength": 4.259, "TrackType": "Permanent", "Corners": 14},
        "Monza": {"TrackLength": 5.793, "TrackType": "Permanent", "Corners": 11},
        "Baku": {"TrackLength": 6.003, "TrackType": "Street", "Corners": 20},
        "Marina_Bay": {"TrackLength": 4.940, "TrackType": "Street", "Corners": 19},
        "Suzuka": {"TrackLength": 5.807, "TrackType": "Permanent", "Corners": 18},
        "Losail": {"TrackLength": 5.380, "TrackType": "Permanent", "Corners": 16},
        "Austin": {"TrackLength": 5.513, "TrackType": "Permanent", "Corners": 20},
        "Mexico": {"TrackLength": 4.304, "TrackType": "Permanent", "Corners": 17},
        "Interlagos": {"TrackLength": 4.309, "TrackType": "Permanent", "Corners": 15},
        "Las_Vegas": {"TrackLength": 6.201, "TrackType": "Street", "Corners": 17},
        "Lusail": {"TrackLength": 5.380, "TrackType": "Permanent", "Corners": 16},
        "Yas_Marina": {"TrackLength": 5.281, "TrackType": "Permanent", "Corners": 16},
        "Jeddah": {"TrackLength": 6.174, "TrackType": "Street", "Corners": 27},
        "Istanbul": {"TrackLength": 5.338, "TrackType": "Permanent", "Corners": 14},
        "Portimão": {"TrackLength": 4.653, "TrackType": "Permanent", "Corners": 15},
        "Mugello": {"TrackLength": 5.245, "TrackType": "Permanent", "Corners": 15},
        "Sochi": {"TrackLength": 5.848, "TrackType": "Street", "Corners": 18},
        "Nürburgring": {"TrackLength": 5.148, "TrackType": "Permanent", "Corners": 15},
    }
    
    # Create "TrackId" from EventName (extracted from RaceId)
    result_df["TrackId"] = result_df["EventName"].str.split("_").str[0]
    
    # Add track information
    for track_id, info in track_info.items():
        mask = result_df["TrackId"].str.contains(track_id, case=False)
        if mask.any():
            for key, value in info.items():
                result_df.loc[mask, key] = value
    
    # Fill missing track information with defaults
    if "TrackLength" not in result_df.columns or result_df["TrackLength"].isna().any():
        result_df["TrackLength"] = result_df["TrackLength"].fillna(5.0)
    
    if "TrackType" not in result_df.columns or result_df["TrackType"].isna().any():
        result_df["TrackType"] = result_df["TrackType"].fillna("Unknown")
    
    if "Corners" not in result_df.columns or result_df["Corners"].isna().any():
        result_df["Corners"] = result_df["Corners"].fillna(15)
    
    # Fill missing values for Boolean columns with False
    boolean_columns = ["SafetyCar", "VirtualSafetyCar", "YellowFlag", "RedFlag", "FreshTyre"]
    for col in boolean_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(False)
    
    # Fill missing values for numerical columns with median by track
    numerical_cols = [
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", 
        "TyreLife", "Stint", "LapDeltaToFastest", "LapDeltaToLeader"
    ]
    
    for col in numerical_cols:
        if col in result_df.columns:
            # Get medians by track
            track_medians = result_df.groupby("TrackId")[col].median()
            
            # Apply medians by track
            for track_id in result_df["TrackId"].unique():
                mask = (result_df["TrackId"] == track_id) & (result_df[col].isna())
                median_value = track_medians.get(track_id, result_df[col].median())
                if not pd.isna(median_value):
                    result_df.loc[mask, col] = median_value
            
            # Fill any remaining NaNs with global median
            global_median = result_df[col].median()
            if not pd.isna(global_median):
                result_df[col] = result_df[col].fillna(global_median)
    
    # Fill missing values for categorical columns
    categorical_cols = ["Compound", "Driver", "Team"]
    for col in categorical_cols:
        if col in result_df.columns and result_df[col].isna().any():
            result_df[col] = result_df[col].fillna("Unknown")
    
    # Filter only race data for the final dataset
    result_df = result_df[result_df["SessionType"] == "R"]
    
    logger.info("Lap data cleaned", rows=len(result_df))
    return result_df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into training, test, and holdout sets.

    Args:
        df: The full dataset.

    Returns:
        Tuple of (train_df, test_df, holdout_df).
    """
    logger.info("Splitting data into train/test/holdout sets")
    
    # Extract year from RaceId field (format: YYYY_RR_EventName_SessionType)
    df["Year"] = df["RaceId"].str.split("_").str[0].astype(int)
    
    # Create holdout set (2024 data)
    holdout_df = df[df["Year"].isin(HOLDOUT_YEARS)]
    
    # Remaining data for train/test split
    non_holdout_df = df[~df["Year"].isin(HOLDOUT_YEARS)]
    
    # Group by race to ensure all laps from the same race are in the same set
    races = non_holdout_df["RaceId"].unique()
    np.random.seed(42)
    test_races = np.random.choice(races, size=int(len(races) * TEST_SIZE), replace=False)
    
    # Create train and test sets
    test_df = non_holdout_df[non_holdout_df["RaceId"].isin(test_races)]
    train_df = non_holdout_df[~non_holdout_df["RaceId"].isin(test_races)]
    
    logger.info(
        "Data split complete",
        train_size=len(train_df),
        test_size=len(test_df),
        holdout_size=len(holdout_df),
    )
    
    return train_df, test_df, holdout_df


def transform_data() -> None:
    """Transform raw F1 data into a processed dataset."""
    logger.info("Starting data transformation")
    
    # Load data
    laps_df = load_all_lap_data()
    track_status_df = load_track_status_data()
    weather_df = load_weather_data()
    
    # Merge data
    merged_df = merge_track_status(laps_df, track_status_df)
    merged_df = merge_weather(merged_df, weather_df)
    
    # Clean data
    cleaned_df = clean_lap_data(merged_df)
    
    # Add safety car target
    df_with_target = add_safety_car_target(cleaned_df)
    
    # Split data
    train_df, test_df, holdout_df = split_train_test(df_with_target)
    
    # Save processed datasets
    save_dataframe(df_with_target, PROCESSED_DATASET_PATH)
    save_dataframe(train_df, TRAIN_DATASET_PATH)
    save_dataframe(test_df, TEST_DATASET_PATH)
    save_dataframe(holdout_df, HOLDOUT_DATASET_PATH)
    
    logger.info(
        "Data transformation complete",
        total_rows=len(df_with_target),
        safety_car_events=df_with_target[TARGET_COLUMN].sum(),
    )


if __name__ == "__main__":
    transform_data() 