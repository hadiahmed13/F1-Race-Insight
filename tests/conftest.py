"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN


@pytest.fixture(scope="session")
def sample_data_path():
    """Get the path to sample data."""
    sample_path = Path("data/processed/samples/f1_safety_car_dataset_sample.parquet")
    
    # Check if sample data exists
    if not sample_path.exists():
        # Try to create sample data
        try:
            from scripts.generate_sample_data import generate_processed_sample
            generate_processed_sample()
        except ImportError:
            pytest.skip("Sample data not available and cannot be generated")
    
    if not sample_path.exists():
        pytest.skip("Sample data not available")
    
    return sample_path


@pytest.fixture(scope="session")
def sample_data(sample_data_path):
    """Load sample data for testing."""
    return pd.read_parquet(sample_data_path)


@pytest.fixture(scope="session")
def sample_model_path():
    """Get the path to sample model."""
    model_path = Path("models/checkpoints/samples/model_sample.joblib")
    pipeline_path = Path("models/checkpoints/samples/pipeline_sample.joblib")
    
    # Check if sample model exists
    if not model_path.exists() or not pipeline_path.exists():
        # Try to create sample model
        try:
            from scripts.generate_sample_model import generate_sample_model
            generate_sample_model()
        except ImportError:
            pytest.skip("Sample model not available and cannot be generated")
    
    if not model_path.exists() or not pipeline_path.exists():
        pytest.skip("Sample model not available")
    
    return model_path, pipeline_path


@pytest.fixture(scope="session")
def sample_model(sample_model_path):
    """Load sample model for testing."""
    model_path, pipeline_path = sample_model_path
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline


@pytest.fixture
def train_test_data(sample_data):
    """Create train/test split of sample data."""
    # Use first 80% for training, last 20% for testing
    df = sample_data.copy()
    n = len(df)
    train_size = int(n * 0.8)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    return train_df, test_df


@pytest.fixture
def features_target_split(train_test_data):
    """Split features and target for training and testing."""
    train_df, test_df = train_test_data
    
    # Get all columns except the target
    feature_cols = [col for col in train_df.columns if col != TARGET_COLUMN]
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]
    
    return X_train, y_train, X_test, y_test 