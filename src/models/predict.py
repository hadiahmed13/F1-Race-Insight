"""Prediction module for F1 Race Insight."""

from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.config import LATEST_MODEL_PATH, PIPELINE_PATH
from src.models.evaluate import get_optimal_threshold
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_model_and_pipeline():
    """Load the latest model and feature pipeline.

    Returns:
        Tuple of (model, pipeline).
    """
    logger.info("Loading model and pipeline")
    
    try:
        model = joblib.load(LATEST_MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        logger.info("Model and pipeline loaded successfully")
        return model, pipeline
    except FileNotFoundError as e:
        logger.error(f"Error loading model or pipeline: {str(e)}")
        raise


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Make batch predictions for safety car deployment.

    Args:
        df: DataFrame with feature data.

    Returns:
        DataFrame with added prediction columns.
    """
    logger.info("Making batch predictions", num_instances=len(df))
    
    # Load model and pipeline
    model, pipeline = load_model_and_pipeline()
    
    # Get optimal threshold
    threshold = get_optimal_threshold()
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Transform features
    X = pipeline.transform(df)
    
    # Make predictions
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Add predictions to the DataFrame
    result_df["sc_probability"] = y_prob
    result_df["sc_prediction"] = y_pred
    result_df["threshold"] = threshold
    
    logger.info(
        "Batch predictions complete",
        positive_pct=f"{100 * y_pred.mean():.2f}%",
    )
    
    return result_df


def predict_single_lap(
    race_id: str, lap_number: int, lap_data: Optional[Dict] = None
) -> Dict:
    """Make a prediction for safety car deployment for a single lap.

    Args:
        race_id: The race identifier (format: YYYY_RR_EventName).
        lap_number: The lap number.
        lap_data: Optional dictionary with lap data. If not provided,
            the function will use defaults or values from similar races.

    Returns:
        Dictionary with prediction results.
    """
    logger.info(
        "Making single lap prediction",
        race_id=race_id,
        lap_number=lap_number,
    )
    
    # Load model and pipeline
    model, pipeline = load_model_and_pipeline()
    
    # Get optimal threshold
    threshold = get_optimal_threshold()
    
    # Create a DataFrame with the single lap
    if lap_data is None:
        lap_data = {}
    
    # Extract year and track from race_id
    try:
        parts = race_id.split("_", 2)
        year = int(parts[0])
        track = parts[2] if len(parts) > 2 else "Unknown"
    except (ValueError, IndexError):
        year = 2024  # Default to current year
        track = "Unknown"
    
    # Create a minimal DataFrame with required features
    # Use sensible defaults for missing values
    single_lap_df = pd.DataFrame(
        {
            "RaceId": [race_id],
            "Year": [year],
            "TrackId": [track],
            "LapNumber": [lap_number],
            "Driver": [lap_data.get("Driver", "Unknown")],
            "Team": [lap_data.get("Team", "Unknown")],
            "Compound": [lap_data.get("Compound", "Unknown")],
            "TyreLife": [lap_data.get("TyreLife", 10)],
            "FreshTyre": [lap_data.get("FreshTyre", False)],
            "Stint": [lap_data.get("Stint", 1)],
            "SpeedI1": [lap_data.get("SpeedI1", 250)],
            "SpeedI2": [lap_data.get("SpeedI2", 250)],
            "SpeedFL": [lap_data.get("SpeedFL", 250)],
            "SpeedST": [lap_data.get("SpeedST", 250)],
            "LapDeltaToFastest": [lap_data.get("LapDeltaToFastest", 0)],
            "LapDeltaToLeader": [lap_data.get("LapDeltaToLeader", 0)],
            "Position": [lap_data.get("Position", 10)],
            "AirTemp": [lap_data.get("AirTemp", 25)],
            "Humidity": [lap_data.get("Humidity", 50)],
            "Rainfall": [lap_data.get("Rainfall", 0)],
            "TrackTemp": [lap_data.get("TrackTemp", 30)],
            "WindSpeed": [lap_data.get("WindSpeed", 10)],
            "VirtualSafetyCar": [lap_data.get("VirtualSafetyCar", False)],
            "YellowFlag": [lap_data.get("YellowFlag", False)],
            "TrackLength": [lap_data.get("TrackLength", 5.0)],
            "Corners": [lap_data.get("Corners", 15)],
        }
    )
    
    # Add interaction features (TyreLife x TrackTemp, etc.)
    single_lap_df["TyreLife_TrackTemp_interaction"] = (
        single_lap_df["TyreLife"] * single_lap_df["TrackTemp"]
    )
    single_lap_df["TrackLength_SpeedST_interaction"] = (
        single_lap_df["TrackLength"] * single_lap_df["SpeedST"]
    )
    single_lap_df["Rainfall_Corners_interaction"] = (
        single_lap_df["Rainfall"] * single_lap_df["Corners"]
    )
    
    # Transform features using the pipeline
    feature_names = list(pipeline.feature_names_in_)
    available_features = [col for col in feature_names if col in single_lap_df.columns]
    
    try:
        X = pipeline.transform(single_lap_df[available_features])
    except Exception as e:
        logger.error(f"Error transforming features: {str(e)}")
        # Fallback approach - try with default values
        logger.info("Using fallback approach with default values")
        X = np.zeros((1, len(model.feature_names_)))
    
    # Make prediction
    y_prob = float(model.predict_proba(X)[0, 1])
    y_pred = int(y_prob >= threshold)
    
    # Prepare result
    result = {
        "race_id": race_id,
        "lap": lap_number,
        "probability": y_prob,
        "threshold": threshold,
        "will_deploy_sc": bool(y_pred),
    }
    
    logger.info(
        "Single lap prediction complete",
        race_id=race_id,
        lap=lap_number,
        probability=y_prob,
        will_deploy_sc=bool(y_pred),
    )
    
    return result 