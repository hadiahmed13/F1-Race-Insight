"""Flask API for F1 Race Insight."""

import datetime
import json
import os
import subprocess
from typing import Dict, Tuple, Union

import flask
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.config import API_CORS_ORIGINS, API_RATE_LIMIT, LATEST_MODEL_PATH
from src.models.predict import predict_single_lap
from src.utils.logging import get_logger

# Create Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": API_CORS_ORIGINS}})

# Configure rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[API_RATE_LIMIT],
    storage_uri="memory://",
)

# Initialize logger
logger = get_logger(__name__)


def get_git_revision() -> str:
    """Get the git revision hash.

    Returns:
        Git revision hash or "unknown" if not available.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def get_model_timestamp() -> str:
    """Get the timestamp of the model file.

    Returns:
        Timestamp of the model file or "unknown" if not available.
    """
    try:
        if os.path.exists(LATEST_MODEL_PATH):
            mtime = os.path.getmtime(LATEST_MODEL_PATH)
            return datetime.datetime.fromtimestamp(mtime).isoformat()
        return "model file not found"
    except Exception:
        return "unknown"


@app.route("/api/v1/health", methods=["GET"])
def health_check() -> Tuple[Dict, int]:
    """Health check endpoint.

    Returns:
        JSON response with status and model information.
    """
    logger.info("Health check requested")
    
    response = {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "git_revision": get_git_revision(),
        "model_timestamp": get_model_timestamp(),
    }
    
    logger.info("Health check successful")
    return jsonify(response), 200


@app.route("/api/v1/predict", methods=["POST"])
@limiter.limit(API_RATE_LIMIT)
def predict() -> Tuple[Dict, int]:
    """Prediction endpoint.

    Returns:
        JSON response with prediction results.
    """
    try:
        # Get request data
        data = request.json
        logger.info("Prediction requested", data=data)
        
        # Extract required parameters
        if not data or "race_id" not in data or "lap" not in data:
            logger.error("Invalid request: missing required parameters")
            return (
                jsonify(
                    {
                        "error": "Invalid request",
                        "message": "Missing required parameters: race_id, lap",
                    }
                ),
                400,
            )
        
        race_id = data["race_id"]
        lap = int(data["lap"])
        
        # Extract optional lap data if provided
        lap_data = data.get("lap_data", None)
        
        # Make prediction
        result = predict_single_lap(race_id, lap, lap_data)
        
        logger.info("Prediction successful", result=result)
        return jsonify(result), 200
    
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        return jsonify({"error": "Invalid value", "message": str(e)}), 400
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )


@app.route("/", methods=["GET"])
def root() -> str:
    """Root endpoint that redirects to API documentation.

    Returns:
        Redirect to API documentation or simple HTML page.
    """
    # A simple HTML page with API information
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>F1 Race Insight API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #E10600;
                border-bottom: 1px solid #ccc;
                padding-bottom: 10px;
            }
            h2 {
                margin-top: 30px;
                color: #15151E;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 5px;
                border-radius: 3px;
            }
            pre {
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <h1>F1 Race Insight API</h1>
        
        <p>This API provides access to the F1 Race Insight safety car prediction model.</p>
        
        <h2>Endpoints</h2>
        
        <h3>1. Health Check</h3>
        <code>GET /api/v1/health</code>
        <p>Returns the current status of the API and model information.</p>
        
        <h3>2. Predict</h3>
        <code>POST /api/v1/predict</code>
        <p>Predicts safety car deployment probability for a specific lap.</p>
        
        <h4>Request Format:</h4>
        <pre>
{
  "race_id": "2024_JPN",
  "lap": 23,
  "lap_data": {
    // Optional additional lap data
    "Driver": "VER",
    "Team": "Red Bull Racing",
    "Compound": "HARD",
    // ... more fields
  }
}
        </pre>
        
        <h4>Response Format:</h4>
        <pre>
{
  "race_id": "2024_JPN",
  "lap": 23,
  "probability": 0.37,
  "threshold": 0.29,
  "will_deploy_sc": true
}
        </pre>
        
        <h2>Rate Limiting</h2>
        <p>This API is rate-limited to 100 requests per minute per IP address.</p>
        
        <hr>
        <p>For more information, visit the <a href="https://github.com/yourusername/f1-race-insight">F1 Race Insight GitHub repository</a>.</p>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 