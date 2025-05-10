"""Global configuration for F1 Race Insight project."""

from pathlib import Path
from typing import Dict, List, Set, Tuple

# Data years to process
DATA_YEARS: List[int] = [2020, 2021, 2022, 2023, 2024]

# Base paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
MODEL_CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"

# Model file paths
LATEST_MODEL_PATH: Path = MODEL_CHECKPOINTS_DIR / "model_latest.joblib"
PIPELINE_PATH: Path = MODEL_CHECKPOINTS_DIR / "pipeline_latest.joblib"

# Dataset file paths
PROCESSED_DATASET_PATH: Path = PROCESSED_DATA_DIR / "f1_safety_car_dataset.parquet"
TRAIN_DATASET_PATH: Path = PROCESSED_DATA_DIR / "train_dataset.parquet"
TEST_DATASET_PATH: Path = PROCESSED_DATA_DIR / "test_dataset.parquet"
HOLDOUT_DATASET_PATH: Path = PROCESSED_DATA_DIR / "holdout_dataset.parquet"

# FastF1 cache settings
FASTF1_CACHE_DIR: Path = RAW_DATA_DIR / "fastf1_cache"

# Track-specific columns
TRACK_COLUMNS: List[str] = [
    "TrackId",
    "TrackLength",
    "TrackType",
    "Corners",
    "TrackStatus",
]

# Driver-specific columns
DRIVER_COLUMNS: List[str] = [
    "DriverNumber",
    "Driver",
    "Team",
    "DriverStatus",
]

# Lap-specific columns
LAP_COLUMNS: List[str] = [
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "LapStartTime",
    "LapStartDate",
    "PitInTime",
    "PitOutTime",
    "PitDuration",
    "IsAccurate",
    "LapDeltaToFastest",
    "LapDeltaToLeader",
    "Position",
]

# Weather-specific columns
WEATHER_COLUMNS: List[str] = [
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "TrackTemp",
    "WindDirection",
    "WindSpeed",
]

# Safety car and race status columns
RACE_STATUS_COLUMNS: List[str] = [
    "SafetyCar",
    "VirtualSafetyCar",
    "YellowFlag",
    "RedFlag",
]

# Target column
TARGET_COLUMN: str = "sc_next_lap"

# Feature columns to use in the model
FEATURE_COLUMNS: List[str] = [
    # Track features
    "TrackLength",
    "Corners",
    
    # Driver features
    "Driver",
    "Team",
    
    # Lap features
    "LapNumber",
    "LapDeltaToFastest",
    "LapDeltaToLeader",
    "Position",
    
    # Tire features
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    
    # Speed features
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    
    # Weather features
    "AirTemp",
    "Humidity",
    "Rainfall",
    "TrackTemp",
    "WindSpeed",
    
    # Race status
    "VirtualSafetyCar",
    "YellowFlag",
]

# Categorical columns
CATEGORICAL_COLUMNS: List[str] = [
    "Driver",
    "Team",
    "Compound",
    "FreshTyre",
]

# Numerical columns
NUMERICAL_COLUMNS: List[str] = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]

# Boolean columns
BOOLEAN_COLUMNS: List[str] = [
    "FreshTyre",
    "VirtualSafetyCar",
    "YellowFlag",
]

# Interaction features to create
INTERACTION_FEATURES: List[Tuple[str, str]] = [
    ("TyreLife", "TrackTemp"),
    ("TrackLength", "SpeedST"),
    ("Rainfall", "Corners"),
]

# API settings
API_RATE_LIMIT = "100 per minute"
API_CORS_ORIGINS: List[str] = ["*"]

# XGBoost model parameters
XGBOOST_PARAMS: Dict = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.02,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

# Cross-validation settings
CV_FOLDS: int = 5
CV_RANDOM_STATE: int = 42

# Train-test split settings
TEST_SIZE: float = 0.2
HOLDOUT_YEARS: Set[int] = {2024}

# Dashboard settings
DASHBOARD_CACHE_TTL: int = 3600  # seconds

# Architecture diagram
# This can be used to generate a simple ASCII diagram
ARCHITECTURE_DIAGRAM = """
+-----------------------------------------------------------+
|                                                           |
|  +-------+    +--------+    +---------+    +-----------+  |
|  | FastF1|    | Raw    |    | ETL     |    | Processed |  |
|  | API   |--->| Data   |--->| Pipeline|--->| Dataset   |  |
|  +-------+    +--------+    +---------+    +-----------+  |
|                                 |               |         |
|                                 v               v         |
|                           +-----------+    +----------+   |
|                           | Feature   |    | Training |   |
|                           | Pipeline  |--->| Pipeline |   |
|                           +-----------+    +----------+   |
|                                                |          |
|                                                v          |
|  +---------+    +---------+    +----------+   |          |
|  | Streamlit|<--| Model   |<---|   Model  |<--+          |
|  | Dashboard|   | Registry|    | Evaluation               |
|  +---------+    +---------+    +----------+              |
|       ^                             ^                    |
|       |                             |                    |
|       v                             v                    |
|  +---------+                  +---------+                |
|  | Flask   |<---------------->| XGBoost |                |
|  | API     |                  | Model   |                |
|  +---------+                  +---------+                |
|       ^                                                  |
|       |                                                  |
|       v                                                  |
|  +---------+                                             |
|  | Client  |                                             |
|  | Apps    |                                             |
|  +---------+                                             |
|                                                           |
+-----------------------------------------------------------+
"""

# Project metadata
PROJECT_NAME = "F1 Race Insight Predictor"
PROJECT_DESCRIPTION = "ML-powered safety car deployment prediction for Formula 1 races"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "F1 Race Insight Team"
PROJECT_LICENSE = "MIT"
PROJECT_REPO = "https://github.com/yourusername/f1-race-insight" 