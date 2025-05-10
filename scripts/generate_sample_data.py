#!/usr/bin/env python
"""Generate sample data for testing and development."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
SAMPLE_SIZE = 100
RAW_SAMPLE_PATH = Path("data/raw/samples/2023_01_Bahrain_R_laps.parquet")
PROCESSED_SAMPLE_PATH = Path("data/processed/samples/f1_safety_car_dataset_sample.parquet")


def generate_raw_sample():
    """Generate a sample of raw lap data."""
    # Ensure directory exists
    os.makedirs(RAW_SAMPLE_PATH.parent, exist_ok=True)
    
    # Generate random lap data
    np.random.seed(42)
    
    # Create 20 drivers
    drivers = [f"Driver{i}" for i in range(1, 21)]
    teams = ["Red Bull", "Ferrari", "Mercedes", "McLaren", "Aston Martin"] * 4
    
    # Create sample dataframe
    data = []
    for lap_number in range(1, 6):  # 5 laps
        for driver_idx, driver in enumerate(drivers):
            # Add some randomness to the data
            lap_time_seconds = 90 + np.random.normal(0, 2)
            
            # Basic lap data
            lap_data = {
                "LapNumber": lap_number,
                "Driver": driver,
                "Team": teams[driver_idx],
                "LapTime": pd.Timedelta(seconds=lap_time_seconds),
                "Compound": np.random.choice(["SOFT", "MEDIUM", "HARD"]),
                "TyreLife": np.random.randint(1, 20),
                "FreshTyre": np.random.choice([True, False]),
                "Stint": np.random.randint(1, 3),
                "Position": driver_idx + 1,
                "IsAccurate": True,
                "LapStartTime": pd.Timestamp("2023-03-05") + pd.Timedelta(minutes=lap_number * 2),
                "SessionType": "R",
                "SpeedI1": np.random.normal(250, 10),
                "SpeedI2": np.random.normal(220, 15),
                "SpeedFL": np.random.normal(270, 8),
                "SpeedST": np.random.normal(300, 12),
                "Year": 2023,
                "Round": 1,
                "EventName": "Bahrain_Grand_Prix",
                "RaceId": "2023_01_Bahrain_R",
            }
            data.append(lap_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to parquet
    df.to_parquet(RAW_SAMPLE_PATH)
    print(f"Raw sample data saved to {RAW_SAMPLE_PATH}")


def generate_processed_sample():
    """Generate a sample of processed data with features and target."""
    # Ensure directory exists
    os.makedirs(PROCESSED_SAMPLE_PATH.parent, exist_ok=True)
    
    # Load raw sample if it exists, otherwise generate it
    if not RAW_SAMPLE_PATH.exists():
        generate_raw_sample()
    
    try:
        df = pd.read_parquet(RAW_SAMPLE_PATH)
    except:
        # If loading fails, regenerate
        generate_raw_sample()
        df = pd.read_parquet(RAW_SAMPLE_PATH)
    
    # Add additional features
    df["TrackId"] = "Bahrain"
    df["TrackLength"] = 5.412
    df["TrackType"] = "Permanent"
    df["Corners"] = 15
    
    # Weather data
    df["AirTemp"] = np.random.normal(28, 1, len(df))
    df["Humidity"] = np.random.normal(45, 5, len(df))
    df["Rainfall"] = 0.0
    df["TrackTemp"] = np.random.normal(35, 2, len(df))
    df["WindSpeed"] = np.random.normal(10, 2, len(df))
    
    # Race status flags
    df["SafetyCar"] = False
    df["VirtualSafetyCar"] = False
    df["YellowFlag"] = False
    df["RedFlag"] = False
    
    # Add one safety car in lap 3
    df.loc[df["LapNumber"] == 3, "SafetyCar"] = True
    
    # Add target variable - safety car on next lap
    df["sc_next_lap"] = 0
    
    # Set sc_next_lap = 1 for lap 2 (since SC is on lap 3)
    df.loc[df["LapNumber"] == 2, "sc_next_lap"] = 1
    
    # Save to parquet
    df.to_parquet(PROCESSED_SAMPLE_PATH)
    print(f"Processed sample data saved to {PROCESSED_SAMPLE_PATH}")


if __name__ == "__main__":
    generate_raw_sample()
    generate_processed_sample() 