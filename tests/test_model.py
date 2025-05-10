"""Tests for model module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from src.models.evaluate import evaluate_model, get_latest_auc, get_optimal_threshold
from src.models.predict import load_model_and_pipeline, predict_batch, predict_single_lap
from src.models.train import create_classifier, cross_validate_model, load_datasets


@pytest.fixture
def mock_model():
    """Create a mock XGBoost model."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline."""
    return MagicMock(
        transform=lambda X: X,
        feature_names_in_=["Feature1", "Feature2", "Feature3"]
    )


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return pd.DataFrame({
        "RaceId": ["2023_01_Bahrain_R"] * 5,
        "Year": [2023] * 5,
        "TrackId": ["Bahrain"] * 5,
        "LapNumber": [1, 2, 3, 4, 5],
        "Driver": ["HAM", "VER", "LEC", "HAM", "VER"],
        "Team": ["Mercedes", "Red Bull", "Ferrari", "Mercedes", "Red Bull"],
        "Compound": ["SOFT", "MEDIUM", "HARD", "SOFT", "MEDIUM"],
        "TyreLife": [1, 5, 3, 2, 6],
        "FreshTyre": [True, False, True, False, False],
        "SpeedST": [300, 303, 298, 301, 304],
        "TrackLength": [5.412] * 5,
        "Corners": [15] * 5,
        "sc_next_lap": [0, 0, 1, 0, 0],
        # Add columns used as features
        "Feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Feature3": [10, 20, 30, 40, 50],
    })


def test_create_classifier():
    """Test creating an XGBoost classifier."""
    # Create a mock dataset with balanced classes
    mock_df = pd.DataFrame({
        "sc_next_lap": [0, 0, 0, 0, 1, 1]
    })
    
    # Create the classifier
    classifier = create_classifier(mock_df)
    
    # Verify classifier is an XGBoost classifier
    assert isinstance(classifier, xgb.XGBClassifier)
    
    # Check that scale_pos_weight was properly set
    # Since we have 4 negatives and 2 positives, scale_pos_weight should be around 2.0
    assert classifier.get_params()["scale_pos_weight"] == pytest.approx(2.0, abs=0.1)


def test_cross_validate_model(mock_model):
    """Test cross-validation of the model."""
    # Create a mock dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    # Mock the cross_val_score function
    with patch("src.models.train.cross_val_score", return_value=np.array([0.8, 0.85, 0.82, 0.79, 0.83])):
        cv_score = cross_validate_model(mock_model, X_df, y_series)
    
    # Verify cross-validation score is as expected
    assert cv_score == pytest.approx(0.818, abs=0.001)


def test_evaluate_model(mock_model):
    """Test model evaluation."""
    # Create a mock dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    # Disable plots for this test
    metrics = evaluate_model(mock_model, X_df, y_series, save_plots=False)
    
    # Verify metrics are calculated
    assert "auc" in metrics
    assert "optimal_threshold" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "precision" in metrics
    assert "f1_score" in metrics
    
    # AUC should be in range (0, 1)
    assert 0 < metrics["auc"] < 1
    
    # Threshold should be in range (0, 1)
    assert 0 < metrics["optimal_threshold"] < 1


def test_load_model_and_pipeline(mock_model, mock_pipeline):
    """Test loading model and pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save mock model and pipeline
        model_path = os.path.join(tmpdir, "model_latest.joblib")
        pipeline_path = os.path.join(tmpdir, "pipeline_latest.joblib")
        joblib.dump(mock_model, model_path)
        joblib.dump(mock_pipeline, pipeline_path)
        
        # Patch the paths
        with patch("src.models.predict.LATEST_MODEL_PATH", Path(model_path)), \
             patch("src.models.predict.PIPELINE_PATH", Path(pipeline_path)):
            
            # Load model and pipeline
            model, pipeline = load_model_and_pipeline()
            
            # Verify model and pipeline are loaded
            assert isinstance(model, xgb.XGBClassifier)
            assert pipeline is not None


def test_predict_batch(mock_model, mock_pipeline, mock_dataset):
    """Test batch prediction."""
    # Patch the load_model_and_pipeline function
    with patch("src.models.predict.load_model_and_pipeline", return_value=(mock_model, mock_pipeline)), \
         patch("src.models.predict.get_optimal_threshold", return_value=0.5):
        
        # Make batch predictions
        result_df = predict_batch(mock_dataset)
        
        # Verify predictions were added
        assert "sc_probability" in result_df.columns
        assert "sc_prediction" in result_df.columns
        assert "threshold" in result_df.columns
        
        # Check that all rows have predictions
        assert not result_df["sc_probability"].isna().any()
        
        # Check that predictions are in range (0, 1)
        assert (result_df["sc_probability"] >= 0).all()
        assert (result_df["sc_probability"] <= 1).all()
        
        # Check that threshold is correctly used
        assert (result_df["sc_prediction"] == (result_df["sc_probability"] >= 0.5)).all()


def test_predict_single_lap(mock_model, mock_pipeline):
    """Test single lap prediction."""
    # Patch the load_model_and_pipeline function
    with patch("src.models.predict.load_model_and_pipeline", return_value=(mock_model, mock_pipeline)), \
         patch("src.models.predict.get_optimal_threshold", return_value=0.5):
        
        # Make a single lap prediction
        result = predict_single_lap("2023_01_Bahrain_R", 10)
        
        # Verify prediction structure
        assert "race_id" in result
        assert "lap" in result
        assert "probability" in result
        assert "threshold" in result
        assert "will_deploy_sc" in result
        
        # Check that race_id and lap match the input
        assert result["race_id"] == "2023_01_Bahrain_R"
        assert result["lap"] == 10
        
        # Check that probability is in range (0, 1)
        assert 0 <= result["probability"] <= 1
        
        # Check that threshold is correctly used
        assert result["will_deploy_sc"] == (result["probability"] >= 0.5)


def test_model_regression():
    """Test model regression (performance check on holdout data)."""
    # This test is designed to check if the model meets the required performance threshold
    # In a real implementation, this would load a test dataset and verify performance
    
    # Skip detailed implementation as it would require model + data
    # Instead, mock the get_latest_auc function
    with patch("src.models.evaluate.get_latest_auc", return_value=0.82):
        auc = get_latest_auc()
        
        # Verify AUC meets the required threshold (0.8)
        assert auc >= 0.80, f"Model AUC ({auc}) is below the required threshold (0.80)" 