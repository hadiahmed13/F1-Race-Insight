"""Feature building module for F1 Race Insight."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS,
                      FEATURE_COLUMNS, INTERACTION_FEATURES,
                      NUMERICAL_COLUMNS)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features from pairs of numerical features.

    Args:
        df: DataFrame with original features.

    Returns:
        DataFrame with added interaction features.
    """
    logger.info("Creating interaction features")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Create interaction terms
    for col1, col2 in INTERACTION_FEATURES:
        if col1 in result_df.columns and col2 in result_df.columns:
            interaction_name = f"{col1}_{col2}_interaction"
            result_df[interaction_name] = result_df[col1] * result_df[col2]
            logger.info(f"Created interaction feature: {interaction_name}")
    
    return result_df


def build_feature_engineering_pipeline() -> Pipeline:
    """Build the feature engineering pipeline.

    Returns:
        Scikit-learn Pipeline for feature engineering.
    """
    logger.info("Building feature engineering pipeline")
    
    # Preprocessors for different column types
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    boolean_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=False)),
        ]
    )
    
    # Filter to ensure columns exist in the data
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in FEATURE_COLUMNS]
    numerical_cols = [col for col in NUMERICAL_COLUMNS if col in FEATURE_COLUMNS]
    boolean_cols = [col for col in BOOLEAN_COLUMNS if col in FEATURE_COLUMNS]
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols),
            ("bool", boolean_transformer, boolean_cols),
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )
    
    # Log pipeline components
    logger.info(
        "Feature engineering pipeline created",
        categorical_columns=categorical_cols,
        numerical_columns=numerical_cols,
        boolean_columns=boolean_cols,
    )
    
    return pipeline


def engineer_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    holdout_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Pipeline]:
    """Apply feature engineering to the dataset.

    Args:
        train_df: Training DataFrame.
        test_df: Optional test DataFrame.
        holdout_df: Optional holdout DataFrame.

    Returns:
        Tuple of (processed_train_df, processed_test_df, processed_holdout_df, pipeline).
    """
    logger.info("Engineering features")
    
    # Add interaction features
    train_df_with_interactions = create_interaction_features(train_df)
    
    test_df_with_interactions = None
    if test_df is not None:
        test_df_with_interactions = create_interaction_features(test_df)
    
    holdout_df_with_interactions = None
    if holdout_df is not None:
        holdout_df_with_interactions = create_interaction_features(holdout_df)
    
    # Add interaction features to the feature columns
    all_feature_columns = FEATURE_COLUMNS.copy()
    for col1, col2 in INTERACTION_FEATURES:
        if col1 in train_df.columns and col2 in train_df.columns:
            all_feature_columns.append(f"{col1}_{col2}_interaction")
    
    # Build and fit the pipeline
    pipeline = build_feature_engineering_pipeline()
    
    # Extract features to use
    X_train = train_df_with_interactions[all_feature_columns]
    
    # Fit the pipeline on training data only
    pipeline.fit(X_train)
    
    # Transform the data
    X_train_transformed = pipeline.transform(X_train)
    
    # Create DataFrame with transformed features
    # Get feature names from OneHotEncoder
    column_names = []
    
    # Get the column transformer and its transformers
    column_transformer = pipeline.named_steps["preprocessor"]
    
    for name, transformer, columns in column_transformer.transformers_:
        if name == "cat":
            # For categorical features, get the onehotencoder output feature names
            encoder = transformer.named_steps["onehot"]
            for i, col in enumerate(columns):
                for category in encoder.categories_[i]:
                    column_names.append(f"{col}_{category}")
        else:
            # For numerical and boolean features, keep original names
            column_names.extend(columns)
    
    # Create DataFrames with transformed features
    train_transformed_df = pd.DataFrame(X_train_transformed, index=train_df.index, columns=column_names)
    train_transformed_df[train_df.columns.difference(all_feature_columns)] = train_df[
        train_df.columns.difference(all_feature_columns)
    ]
    
    test_transformed_df = None
    if test_df is not None:
        X_test = test_df_with_interactions[all_feature_columns]
        X_test_transformed = pipeline.transform(X_test)
        test_transformed_df = pd.DataFrame(X_test_transformed, index=test_df.index, columns=column_names)
        test_transformed_df[test_df.columns.difference(all_feature_columns)] = test_df[
            test_df.columns.difference(all_feature_columns)
        ]
    
    holdout_transformed_df = None
    if holdout_df is not None:
        X_holdout = holdout_df_with_interactions[all_feature_columns]
        X_holdout_transformed = pipeline.transform(X_holdout)
        holdout_transformed_df = pd.DataFrame(
            X_holdout_transformed, index=holdout_df.index, columns=column_names
        )
        holdout_transformed_df[holdout_df.columns.difference(all_feature_columns)] = holdout_df[
            holdout_df.columns.difference(all_feature_columns)
        ]
    
    logger.info(
        "Feature engineering complete",
        train_shape=train_transformed_df.shape,
        test_shape=test_transformed_df.shape if test_transformed_df is not None else None,
        holdout_shape=holdout_transformed_df.shape if holdout_transformed_df is not None else None,
        feature_count=len(column_names),
    )
    
    return train_transformed_df, test_transformed_df, holdout_transformed_df, pipeline 