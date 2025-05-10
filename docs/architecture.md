# F1 Race Insight Architecture

This document describes the architecture of the F1 Race Insight system, which predicts safety car deployments in Formula 1 races.

## Component Diagram

```
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
```

## System Components

### Data Collection

- **FastF1 API**: External Python library for accessing Formula 1 telemetry data
  - Fetches data for qualifying and race sessions
  - Caches data locally to avoid redundant downloads
  - Provides lap-by-lap telemetry, car setups, weather, and track status

- **Raw Data Storage**: Parquet files containing raw data from FastF1
  - Located in `/data/raw` directory
  - Organized by season, round, and session type
  - Contains telemetry, weather, car setup, and track status data

### ETL Pipeline

- **ETL Downloader**: Downloads and caches F1 telemetry data
  - Incrementally downloads new races as they become available
  - Handles retries and error handling for API failures
  - Stores data in a standardized format for processing

- **ETL Transformer**: Processes raw data into a feature-rich dataset
  - Merges different data sources (laps, weather, track status)
  - Cleans data by handling missing values and outliers
  - Creates a target variable for safety car deployment
  - Splits data into training, test, and holdout sets

### Feature Engineering

- **Feature Pipeline**: Transforms raw features into model-ready format
  - Handles one-hot encoding for categorical variables
  - Normalizes numerical features with StandardScaler
  - Creates interaction features (e.g., tyre age × track temperature)
  - Applies feature selection to improve model performance

### Model Training

- **Training Pipeline**: Trains an XGBoost model to predict safety car deployments
  - Uses stratified k-fold cross-validation
  - Optimizes hyperparameters for maximum AUC ROC
  - Applies class weighting to handle imbalanced data
  - Saves the trained model to the model registry

- **Model Evaluation**: Evaluates model performance
  - Calculates AUC ROC, precision, recall, and F1 score
  - Generates performance plots (ROC curve, PR curve, calibration plot)
  - Creates SHAP explanations for feature importance
  - Verifies that performance meets the ≥0.81 AUC threshold

### Model Serving

- **Model Registry**: Stores trained models and their metadata
  - Uses joblib for serialization
  - Tracks version history and performance metrics
  - Facilitates model A/B testing and rollbacks

- **Flask API**: Serves predictions through a REST API
  - Provides `/api/v1/predict` endpoint for safety car predictions
  - Implements `/api/v1/health` endpoint for health checks
  - Uses rate limiting to prevent abuse
  - Includes CORS support for cross-origin requests

### Visualization

- **Streamlit Dashboard**: Interactive web dashboard for exploring predictions
  - Displays lap-by-lap safety car risk
  - Shows track map with risk annotations
  - Provides SHAP explanations for individual predictions
  - Allows race and lap selection through a sidebar

### Deployment & Operations

- **GitHub Actions**: CI/CD automation
  - Runs tests and linting on pull requests
  - Performs scheduled ETL and model retraining
  - Deploys new models to production if they meet performance criteria

- **Heroku**: Cloud deployment platform
  - Hosts the API and dashboard
  - Scales automatically based on load
  - Provides logging and monitoring

## Data Flow

1. The ETL pipeline fetches data from the FastF1 API and stores it in the raw data storage.
2. The transformer processes the raw data into a feature-rich dataset suitable for machine learning.
3. The feature pipeline further transforms the data for model training.
4. The training pipeline trains an XGBoost model on the processed data.
5. The model evaluation component validates that the model meets performance criteria.
6. The model is saved to the model registry.
7. The Flask API loads the model from the registry to serve predictions.
8. The Streamlit dashboard uses the same model to visualize predictions.
9. Client applications can access predictions through the REST API.

## Technical Decisions

- **Parquet Format**: Used for data storage due to its columnar nature, compression, and schema enforcement.
- **XGBoost**: Chosen for its performance on tabular data, robustness to missing values, and feature importance capabilities.
- **Flask**: Used for the API due to its simplicity, performance, and wide adoption.
- **Streamlit**: Selected for dashboard development for its rapid prototyping capabilities and interactive features.
- **GitHub Actions**: Used for CI/CD automation to ensure code quality and automate model retraining.
- **Docker**: Used for containerization to ensure consistent deployment across environments.

## Performance Considerations

- The ETL pipeline is designed to run incrementally, only processing new data.
- Caching is used extensively throughout the system to minimize computation.
- API responses are optimized for low latency.
- The model is preloaded in the API to avoid cold starts.
- Dashboard visualizations are cached to improve user experience. 