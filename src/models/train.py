"""Model training module for F1 Race Insight."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import (CV_FOLDS, CV_RANDOM_STATE, HOLDOUT_DATASET_PATH,
                      LATEST_MODEL_PATH, PIPELINE_PATH, PROCESSED_DATASET_PATH,
                      TARGET_COLUMN, TEST_DATASET_PATH, TRAIN_DATASET_PATH,
                      XGBOOST_PARAMS)
from src.features.build_features import engineer_features
from src.models.evaluate import evaluate_model
from src.utils.io import ensure_dir_exists, load_dataframe
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and holdout datasets.

    Returns:
        Tuple of (train_df, test_df, holdout_df).
    """
    logger.info("Loading datasets")
    
    # Check if split datasets exist
    if (
        os.path.exists(TRAIN_DATASET_PATH)
        and os.path.exists(TEST_DATASET_PATH)
        and os.path.exists(HOLDOUT_DATASET_PATH)
    ):
        logger.info("Loading pre-split datasets")
        train_df = load_dataframe(TRAIN_DATASET_PATH)
        test_df = load_dataframe(TEST_DATASET_PATH)
        holdout_df = load_dataframe(HOLDOUT_DATASET_PATH)
    
    # Otherwise load the full dataset (should not happen in normal operation)
    else:
        logger.warning("Pre-split datasets not found, loading full dataset")
        full_df = load_dataframe(PROCESSED_DATASET_PATH)
        
        # Split by race as a fallback
        all_races = full_df["RaceId"].unique()
        np.random.seed(42)
        np.random.shuffle(all_races)
        
        # For sample data, we may only have one race, so split by rows
        if len(all_races) <= 1:
            # Split by rows
            indices = np.arange(len(full_df))
            np.random.shuffle(indices)
            
            # Use 70% for train, 20% for test, 10% for holdout
            train_idx = indices[:int(0.7 * len(indices))]
            test_idx = indices[int(0.7 * len(indices)):int(0.9 * len(indices))]
            holdout_idx = indices[int(0.9 * len(indices)):]
            
            train_df = full_df.iloc[train_idx].copy()
            test_df = full_df.iloc[test_idx].copy()
            holdout_df = full_df.iloc[holdout_idx].copy()
        else:
            # Normal case - split by races
            train_races = all_races[:int(0.7 * len(all_races))]
            test_races = all_races[int(0.7 * len(all_races)):int(0.9 * len(all_races))]
            holdout_races = all_races[int(0.9 * len(all_races)):]
            
            train_df = full_df[full_df["RaceId"].isin(train_races)]
            test_df = full_df[full_df["RaceId"].isin(test_races)]
            holdout_df = full_df[full_df["RaceId"].isin(holdout_races)]
    
    logger.info(
        "Datasets loaded",
        train_shape=train_df.shape,
        test_shape=test_df.shape,
        holdout_shape=holdout_df.shape,
    )
    
    return train_df, test_df, holdout_df


def create_classifier(train_df: pd.DataFrame) -> xgb.XGBClassifier:
    """Create and configure an XGBoost classifier.

    Args:
        train_df: Training DataFrame, used to compute class weights.

    Returns:
        Configured XGBClassifier.
    """
    logger.info("Creating XGBoost classifier")
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df[TARGET_COLUMN]),
        y=train_df[TARGET_COLUMN],
    )
    
    # Create weight dictionary (class 0: weight 0, class 1: weight 1)
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Create classifier with parameters from config
    classifier = xgb.XGBClassifier(
        **XGBOOST_PARAMS,
        scale_pos_weight=weight_dict[1] / weight_dict[0],  # Ratio of weights
    )
    
    logger.info(
        "XGBoost classifier created",
        params=classifier.get_params(),
        class_weights=weight_dict,
    )
    
    return classifier


def cross_validate_model(
    classifier: xgb.XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    """Perform cross-validation of the model.

    Args:
        classifier: The XGBoost classifier.
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Mean cross-validation AUC score.
    """
    logger.info("Performing cross-validation")
    
    # Create CV splitter
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    # Run cross-validation
    cv_scores = cross_val_score(
        classifier, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    
    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()
    
    logger.info(
        "Cross-validation completed",
        cv_scores=cv_scores.tolist(),
        mean_auc=mean_cv_score,
        std_auc=std_cv_score,
    )
    
    return mean_cv_score


def train_model() -> None:
    """Train and save the safety car prediction model.

    This function:
    1. Loads the datasets
    2. Engineers features
    3. Creates and fits the XGBoost classifier
    4. Evaluates the model on test and holdout sets
    5. Saves the model and pipeline
    """
    logger.info("Starting model training")
    
    # Load datasets
    train_df, test_df, holdout_df = load_datasets()
    
    # Engineer features
    logger.info("Engineering features for model training")
    (
        train_transformed_df,
        test_transformed_df,
        holdout_transformed_df,
        pipeline,
    ) = engineer_features(train_df, test_df, holdout_df)
    
    # Prepare feature matrix and target vector
    X_train = train_transformed_df.drop(columns=[TARGET_COLUMN])
    y_train = train_transformed_df[TARGET_COLUMN]
    
    X_test = test_transformed_df.drop(columns=[TARGET_COLUMN])
    y_test = test_transformed_df[TARGET_COLUMN]
    
    X_holdout = holdout_transformed_df.drop(columns=[TARGET_COLUMN])
    y_holdout = holdout_transformed_df[TARGET_COLUMN]
    
    # Create classifier
    classifier = create_classifier(train_df)
    
    # Cross-validate model
    cv_score = cross_validate_model(classifier, X_train, y_train)
    
    # Train the final model on all training data
    logger.info("Training final model on all training data")
    classifier.fit(X_train, y_train)
    
    # Evaluate model on test and holdout sets
    logger.info("Evaluating model on test set")
    test_metrics = evaluate_model(classifier, X_test, y_test, "test")
    
    logger.info("Evaluating model on holdout set")
    holdout_metrics = evaluate_model(classifier, X_holdout, y_holdout, "holdout")
    
    # Check if model meets performance criteria
    holdout_auc = holdout_metrics["auc"]
    if holdout_auc < 0.8:
        logger.warning(
            f"Model performance below threshold on holdout set: {holdout_auc:.4f} < 0.80"
        )
    else:
        logger.info(f"Model meets performance criteria: {holdout_auc:.4f} >= 0.80")
    
    # Save the model and pipeline
    ensure_dir_exists(LATEST_MODEL_PATH.parent)
    
    logger.info(f"Saving model to {LATEST_MODEL_PATH}")
    joblib.dump(classifier, LATEST_MODEL_PATH)
    
    logger.info(f"Saving feature pipeline to {PIPELINE_PATH}")
    joblib.dump(pipeline, PIPELINE_PATH)
    
    logger.info(
        "Model training complete",
        cv_score=cv_score,
        test_auc=test_metrics["auc"],
        holdout_auc=holdout_metrics["auc"],
    )


if __name__ == "__main__":
    train_model() 