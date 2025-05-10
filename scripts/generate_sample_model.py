#!/usr/bin/env python
"""Generate a sample model for testing and development."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constants
PROCESSED_SAMPLE_PATH = Path("data/processed/samples/f1_safety_car_dataset_sample.parquet")
MODEL_SAMPLE_PATH = Path("models/checkpoints/samples/model_sample.joblib")
PIPELINE_SAMPLE_PATH = Path("models/checkpoints/samples/pipeline_sample.joblib")

# Feature columns to use in the model
FEATURE_COLUMNS = [
    "TrackLength",
    "Corners",
    "Driver",
    "Team",
    "LapNumber",
    "Position",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "AirTemp",
    "TrackTemp",
    "YellowFlag",
]

# Categorical columns
CATEGORICAL_COLUMNS = ["Driver", "Team", "Compound", "FreshTyre"]

# Numerical columns
NUMERICAL_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]

# Target column
TARGET_COLUMN = "sc_next_lap"


def generate_sample_model():
    """Generate a sample XGBoost model and preprocessing pipeline."""
    # Ensure directories exist
    os.makedirs(MODEL_SAMPLE_PATH.parent, exist_ok=True)
    
    # Check if the processed sample data exists
    if not PROCESSED_SAMPLE_PATH.exists():
        print(f"Processed sample data not found at {PROCESSED_SAMPLE_PATH}")
        print("Running generate_sample_data.py script...")
        from generate_sample_data import generate_processed_sample
        generate_processed_sample()
    
    # Load the processed sample data
    df = pd.read_parquet(PROCESSED_SAMPLE_PATH)
    
    # Create preprocessing pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
            ("num", numerical_transformer, NUMERICAL_COLUMNS),
        ]
    )
    
    # Extract features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Fit the preprocessing pipeline
    X_transformed = preprocessor.fit_transform(X)
    
    # Create and train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,  # Small model for sample
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
    )
    
    model.fit(X_transformed, y)
    
    # Save the model and preprocessing pipeline
    joblib.dump(model, MODEL_SAMPLE_PATH)
    joblib.dump(preprocessor, PIPELINE_SAMPLE_PATH)
    
    print(f"Sample model saved to {MODEL_SAMPLE_PATH}")
    print(f"Sample preprocessing pipeline saved to {PIPELINE_SAMPLE_PATH}")
    
    # Create a test prediction to verify the model works
    test_row = X.iloc[0:1]
    test_transformed = preprocessor.transform(test_row)
    prediction = model.predict_proba(test_transformed)
    
    print(f"Test prediction (probability of safety car): {prediction[0][1]:.4f}")


if __name__ == "__main__":
    generate_sample_model() 