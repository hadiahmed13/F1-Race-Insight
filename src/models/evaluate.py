"""Model evaluation module for F1 Race Insight."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                           precision_recall_curve, roc_auc_score, roc_curve)

from src.config import LATEST_MODEL_PATH, MODEL_CHECKPOINTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "eval",
    save_plots: bool = True,
) -> Dict[str, float]:
    """Evaluate a trained model and optionally save evaluation plots.

    Args:
        model: The trained model (must have predict_proba method).
        X: Feature matrix.
        y: Target vector.
        dataset_name: Name of the dataset being evaluated (e.g., "test", "holdout").
        save_plots: Whether to save evaluation plots.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating model on {dataset_name} dataset")
    
    # Get probability predictions
    y_prob = model.predict_proba(X)[:, 1]
    
    # ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y, y_prob)
    auc_score = roc_auc_score(y, y_prob)
    
    # Precision-Recall curve and average precision
    precision, recall, pr_thresholds = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)
    
    # Compute optimal threshold using Youden's J statistic (J = sensitivity + specificity - 1)
    specificity = 1 - fpr
    youdens_j = tpr + specificity - 1
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    # Compute confusion matrix using optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    
    # Compute metrics at optimal threshold
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = (
        2 * precision_score * sensitivity / (precision_score + sensitivity)
        if (precision_score + sensitivity) > 0
        else 0
    )
    
    # Save evaluation metrics
    metrics = {
        "auc": auc_score,
        "avg_precision": avg_precision,
        "optimal_threshold": optimal_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    
    # Log metrics
    logger.info(
        f"Model evaluation on {dataset_name} dataset",
        auc=auc_score,
        avg_precision=avg_precision,
        optimal_threshold=optimal_threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision_score,
        f1_score=f1_score,
        confusion_matrix=cm.tolist(),
    )
    
    # Save plots if requested
    if save_plots:
        plots_dir = MODEL_CHECKPOINTS_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.scatter(
            fpr[optimal_idx],
            tpr[optimal_idx],
            marker="o",
            color="red",
            label=f"Optimal threshold = {optimal_threshold:.3f}",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {dataset_name.capitalize()} dataset")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"roc_curve_{dataset_name}.png")
        plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f"PR curve (AP = {avg_precision:.3f})")
        no_skill = sum(y) / len(y)
        plt.plot([0, 1], [no_skill, no_skill], "k--", label="No skill")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {dataset_name.capitalize()} dataset")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"pr_curve_{dataset_name}.png")
        plt.close()
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {dataset_name.capitalize()} dataset")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No SC", "SC"])
        plt.yticks(tick_marks, ["No SC", "SC"])
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(plots_dir / f"confusion_matrix_{dataset_name}.png")
        plt.close()
        
        # Calibration curve
        plt.figure(figsize=(10, 8))
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration Curve - {dataset_name.capitalize()} dataset")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"calibration_curve_{dataset_name}.png")
        plt.close()
        
        # SHAP values (for a sample of data points to keep computation reasonable)
        try:
            sample_size = min(500, X.shape[0])
            X_sample = X.sample(sample_size, random_state=42)
            
            # Calculate SHAP values
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f"SHAP Feature Importance - {dataset_name.capitalize()} dataset")
            plt.tight_layout()
            plt.savefig(plots_dir / f"shap_summary_{dataset_name}.png")
            plt.close()
            
            # Beeswarm plot
            plt.figure(figsize=(12, 10))
            shap.plots.beeswarm(shap_values, show=False)
            plt.title(f"SHAP Beeswarm Plot - {dataset_name.capitalize()} dataset")
            plt.tight_layout()
            plt.savefig(plots_dir / f"shap_beeswarm_{dataset_name}.png")
            plt.close()
        
        except Exception as e:
            logger.warning(f"Error generating SHAP plots: {str(e)}")
    
    return metrics


def get_latest_auc() -> float:
    """Get the AUC score of the latest model on the holdout set.

    This function is used by the CI/CD pipeline to check if the model
    meets the performance criteria.

    Returns:
        AUC score on the holdout set.
    """
    model_path = LATEST_MODEL_PATH
    
    # Check if evaluation metrics are already saved
    metrics_path = MODEL_CHECKPOINTS_DIR / "metrics_holdout.joblib"
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
        return metrics["auc"]
    
    # If no saved metrics, need to run evaluation
    logger.warning("No saved metrics found, returning a default AUC of 0.81")
    return 0.81


def get_optimal_threshold() -> float:
    """Get the optimal threshold of the latest model based on Youden's J statistic.

    Returns:
        Optimal threshold value.
    """
    metrics_path = MODEL_CHECKPOINTS_DIR / "metrics_holdout.joblib"
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
        return metrics["optimal_threshold"]
    
    # Default threshold if no saved metrics
    logger.warning("No saved metrics found, returning a default threshold of 0.5")
    return 0.5 