"""Tests for API endpoints."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from flask import Flask
from flask.testing import FlaskClient

from src.api.app import app
from src.models.predict import predict_single_lap


@pytest.fixture
def client() -> FlaskClient:
    """Create a test client for the API."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_prediction_result():
    """Create a mock prediction result."""
    return {
        "race_id": "2023_01_Bahrain_R",
        "lap": 10,
        "probability": 0.35,
        "threshold": 0.5,
        "will_deploy_sc": False,
    }


def test_health_endpoint(client: FlaskClient):
    """Test the health endpoint."""
    response = client.get("/api/v1/health")
    
    # Check response status code
    assert response.status_code == 200
    
    # Parse JSON response
    data = json.loads(response.data)
    
    # Check response structure
    assert "status" in data
    assert "timestamp" in data
    assert "git_revision" in data
    assert "model_timestamp" in data
    
    # Check status is OK
    assert data["status"] == "ok"


def test_predict_endpoint_valid_request(client: FlaskClient, mock_prediction_result):
    """Test the predict endpoint with a valid request."""
    # Mock the predict_single_lap function
    with patch(
        "src.api.app.predict_single_lap", return_value=mock_prediction_result
    ) as mock_predict:
        # Make a request
        response = client.post(
            "/api/v1/predict",
            json={"race_id": "2023_01_Bahrain_R", "lap": 10},
            content_type="application/json",
        )
        
        # Check that predict_single_lap was called with correct arguments
        mock_predict.assert_called_once_with("2023_01_Bahrain_R", 10, None)
        
        # Check response status code
        assert response.status_code == 200
        
        # Parse JSON response
        data = json.loads(response.data)
        
        # Check response structure
        assert "race_id" in data
        assert "lap" in data
        assert "probability" in data
        assert "threshold" in data
        assert "will_deploy_sc" in data
        
        # Check values match mock result
        assert data["race_id"] == mock_prediction_result["race_id"]
        assert data["lap"] == mock_prediction_result["lap"]
        assert data["probability"] == mock_prediction_result["probability"]
        assert data["threshold"] == mock_prediction_result["threshold"]
        assert data["will_deploy_sc"] == mock_prediction_result["will_deploy_sc"]


def test_predict_endpoint_missing_parameters(client: FlaskClient):
    """Test the predict endpoint with missing parameters."""
    # Test with missing race_id
    response = client.post(
        "/api/v1/predict",
        json={"lap": 10},
        content_type="application/json",
    )
    
    # Check response status code
    assert response.status_code == 400
    
    # Parse JSON response
    data = json.loads(response.data)
    
    # Check error message
    assert "error" in data
    assert "message" in data
    assert "missing required parameters" in data["message"].lower()
    
    # Test with missing lap
    response = client.post(
        "/api/v1/predict",
        json={"race_id": "2023_01_Bahrain_R"},
        content_type="application/json",
    )
    
    # Check response status code
    assert response.status_code == 400


def test_predict_endpoint_with_lap_data(client: FlaskClient, mock_prediction_result):
    """Test the predict endpoint with additional lap data."""
    lap_data = {
        "Driver": "HAM",
        "Team": "Mercedes",
        "Compound": "SOFT",
        "TyreLife": 5,
    }
    
    # Mock the predict_single_lap function
    with patch(
        "src.api.app.predict_single_lap", return_value=mock_prediction_result
    ) as mock_predict:
        # Make a request
        response = client.post(
            "/api/v1/predict",
            json={
                "race_id": "2023_01_Bahrain_R",
                "lap": 10,
                "lap_data": lap_data,
            },
            content_type="application/json",
        )
        
        # Check that predict_single_lap was called with correct arguments
        mock_predict.assert_called_once_with("2023_01_Bahrain_R", 10, lap_data)
        
        # Check response status code
        assert response.status_code == 200


def test_predict_endpoint_internal_error(client: FlaskClient):
    """Test the predict endpoint when an internal error occurs."""
    # Mock the predict_single_lap function to raise an exception
    with patch(
        "src.api.app.predict_single_lap", side_effect=Exception("Test exception")
    ):
        # Make a request
        response = client.post(
            "/api/v1/predict",
            json={"race_id": "2023_01_Bahrain_R", "lap": 10},
            content_type="application/json",
        )
        
        # Check response status code
        assert response.status_code == 500
        
        # Parse JSON response
        data = json.loads(response.data)
        
        # Check error message
        assert "error" in data
        assert "message" in data
        assert "internal server error" in data["error"].lower()
        assert "test exception" in data["message"].lower()


def test_root_endpoint(client: FlaskClient):
    """Test the root endpoint."""
    response = client.get("/")
    
    # Check response status code
    assert response.status_code == 200
    
    # Check that response is HTML
    assert b"<!DOCTYPE html>" in response.data
    assert b"F1 Race Insight API" in response.data
    
    # Check that it contains API documentation
    assert b"Endpoints" in response.data
    assert b"/api/v1/health" in response.data
    assert b"/api/v1/predict" in response.data 