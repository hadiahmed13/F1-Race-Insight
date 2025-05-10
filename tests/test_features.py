"""Tests for feature modules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.build_features import (build_feature_engineering_pipeline,
                                      create_interaction_features,
                                      engineer_features)
from src.features.target import add_safety_car_target


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing features."""
    return pd.DataFrame({
        "RaceId": ["2023_01_Bahrain_R"] * 5,
        "LapNumber": [1, 2, 3, 4, 5],
        "Driver": ["HAM", "VER", "LEC", "HAM", "VER"],
        "Team": ["Mercedes", "Red Bull", "Ferrari", "Mercedes", "Red Bull"],
        "Compound": ["SOFT", "MEDIUM", "HARD", "SOFT", "MEDIUM"],
        "TyreLife": [1, 5, 3, 2, 6],
        "FreshTyre": [True, False, True, False, False],
        "Position": [3, 1, 2, 4, 1],
        "LapDeltaToFastest": [0.5, 0.0, 0.3, 0.7, 0.1],
        "LapDeltaToLeader": [1.2, 0.0, 0.8, 1.5, 0.0],
        "SpeedI1": [280, 282, 279, 281, 283],
        "SpeedI2": [240, 243, 238, 241, 244],
        "SpeedFL": [270, 273, 268, 271, 274],
        "SpeedST": [300, 303, 298, 301, 304],
        "AirTemp": [28.5, 28.5, 28.6, 28.6, 28.7],
        "Humidity": [45.0, 45.0, 44.5, 44.5, 44.0],
        "Rainfall": [0.0, 0.0, 0.0, 0.0, 0.0],
        "TrackTemp": [36.0, 36.0, 36.5, 36.5, 37.0],
        "WindSpeed": [10.0, 10.0, 10.5, 10.5, 11.0],
        "TrackLength": [5.412, 5.412, 5.412, 5.412, 5.412],
        "Corners": [15, 15, 15, 15, 15],
        "VirtualSafetyCar": [False, False, False, False, False],
        "YellowFlag": [False, False, True, False, False],
        "SafetyCar": [False, False, False, True, False],
    })


def test_add_safety_car_target(sample_df):
    """Test adding safety car target."""
    result_df = add_safety_car_target(sample_df)
    
    # Verify that target column was added
    assert "sc_next_lap" in result_df.columns
    
    # Check that lap 2 has target=1 (safety car in lap 3)
    assert result_df.loc[result_df["LapNumber"] == 3, "sc_next_lap"].iloc[0] == 1
    
    # Check that other laps have target=0
    assert result_df.loc[result_df["LapNumber"] == 1, "sc_next_lap"].iloc[0] == 0
    assert result_df.loc[result_df["LapNumber"] == 2, "sc_next_lap"].iloc[0] == 0
    assert result_df.loc[result_df["LapNumber"] == 4, "sc_next_lap"].iloc[0] == 0
    assert result_df.loc[result_df["LapNumber"] == 5, "sc_next_lap"].iloc[0] == 0


def test_create_interaction_features(sample_df):
    """Test creating interaction features."""
    # Create interaction features
    result_df = create_interaction_features(sample_df)
    
    # Check that interaction features were created
    assert "TyreLife_TrackTemp_interaction" in result_df.columns
    assert "TrackLength_SpeedST_interaction" in result_df.columns
    assert "Rainfall_Corners_interaction" in result_df.columns
    
    # Verify correct calculation of interaction features
    # TyreLife * TrackTemp
    assert result_df.loc[0, "TyreLife_TrackTemp_interaction"] == 1 * 36.0
    assert result_df.loc[1, "TyreLife_TrackTemp_interaction"] == 5 * 36.0
    
    # TrackLength * SpeedST
    assert result_df.loc[0, "TrackLength_SpeedST_interaction"] == 5.412 * 300
    assert result_df.loc[1, "TrackLength_SpeedST_interaction"] == 5.412 * 303
    
    # Rainfall * Corners
    assert result_df.loc[0, "Rainfall_Corners_interaction"] == 0.0 * 15
    assert result_df.loc[1, "Rainfall_Corners_interaction"] == 0.0 * 15


def test_build_feature_engineering_pipeline():
    """Test building the feature engineering pipeline."""
    pipeline = build_feature_engineering_pipeline()
    
    # Check that pipeline is created
    assert isinstance(pipeline, Pipeline)
    
    # Check that preprocessor is included
    assert "preprocessor" in pipeline.named_steps
    
    # Get the column transformer
    column_transformer = pipeline.named_steps["preprocessor"]
    
    # Check that all transformers are included
    transformer_names = [name for name, _, _ in column_transformer.transformers]
    assert "cat" in transformer_names
    assert "num" in transformer_names
    assert "bool" in transformer_names


def test_engineer_features(sample_df):
    """Test engineering features on a dataset."""
    # Add target column
    sample_df = add_safety_car_target(sample_df)
    
    # Engineer features
    train_df, _, _, pipeline = engineer_features(sample_df)
    
    # Check that pipeline was created
    assert pipeline is not None
    
    # Check that transformed data has expected number of rows
    assert len(train_df) == len(sample_df)
    
    # Check that target column is preserved
    assert "sc_next_lap" in train_df.columns
    
    # Check that non-feature columns are preserved
    assert "RaceId" in train_df.columns
    assert "SafetyCar" in train_df.columns
    
    # Check that categorical columns are one-hot encoded
    # For example, check if there are columns for Driver_HAM, Team_Mercedes, etc.
    categorical_cols = ["Driver", "Team", "Compound", "FreshTyre"]
    for col in categorical_cols:
        # At least one column should start with the original column name
        assert any(c.startswith(f"{col}_") for c in train_df.columns) 