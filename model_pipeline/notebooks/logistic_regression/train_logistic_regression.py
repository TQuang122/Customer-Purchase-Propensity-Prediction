#!/usr/bin/env python3
"""
Logistic Regression Baseline Training Script
Customer Purchase Propensity Prediction

This script trains a Logistic Regression model following the plan:
1. Load data from Feast Feature Store (parquet file)
2. Preprocessing: StandardScaler for numerical, OneHotEncoder for categorical
3. Train/Val/Test split: 64%/16%/20%
4. Regularization tuning on validation set
5. Evaluate with Accuracy, Precision, Recall, F1, AUC-ROC
6. Save metrics to JSON
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE TRAINING")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading data from Feast Feature Store...")

    script_dir = Path(__file__).parent
    parquet_path = (
        script_dir
        / "../../data_pipeline/propensity_feature_store/propensity_features/feature_repo/data/processed_purchase_propensity_data_v1.parquet"
    )
    parquet_path = parquet_path.resolve()

    print(f"Loading from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # -------------------------------------------------------------------------
    # 2. Define Features and Preprocessing
    # -------------------------------------------------------------------------
    print("\n[2/6] Preparing features and preprocessing pipeline...")

    NUMERICAL_FEATURES = ["price", "activity_count", "event_weekday"]
    CATEGORICAL_FEATURES = ["brand", "category_code_level1", "category_code_level2"]
    TARGET = "is_purchased"
    ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    print(f"Numerical features: {NUMERICAL_FEATURES}")
    print(f"Categorical features: {CATEGORICAL_FEATURES}")
    print(f"Target: {TARGET}")

    # Prepare X and y
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    # Convert categorical columns to string type
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)

    print(f"\nTarget distribution:")
    print(
        f"  Class 0 (Not Purchased): {(y == 0).sum():,} ({(y == 0).mean() * 100:.2f}%)"
    )
    print(
        f"  Class 1 (Purchased):     {(y == 1).sum():,} ({(y == 1).mean() * 100:.2f}%)"
    )

    # -------------------------------------------------------------------------
    # 3. Train/Validation/Test Split (64%/16%/20%)
    # -------------------------------------------------------------------------
    print("\n[3/6] Splitting data (64%/16%/20%)...")

    # First split: 80% train+val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: 80% train, 20% val (of the 80% = 64% and 16% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    print(
        f"Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0] / len(X) * 100:.1f}%)"
    )
    print(
        f"Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0] / len(X) * 100:.1f}%)"
    )
    print(
        f"Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0] / len(X) * 100:.1f}%)"
    )

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, max_categories=100
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # -------------------------------------------------------------------------
    # 4. Regularization Tuning on Validation Set
    # -------------------------------------------------------------------------
    print("\n[4/6] Tuning regularization parameter C...")
    print("-" * 50)

    C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
    tuning_results = []

    for C in C_VALUES:
        start_time = time.time()
        print(f"\nTraining with C={C}...", end=" ", flush=True)

        # Create pipeline
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        solver="lbfgs",
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        # Fit on training data
        pipeline.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

        # Calculate metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average="macro")
        val_auc = roc_auc_score(y_val, y_val_proba)

        elapsed = time.time() - start_time

        result = {
            "C": C,
            "accuracy": val_accuracy,
            "f1_macro": val_f1,
            "auc_roc": val_auc,
            "pipeline": pipeline,
        }
        tuning_results.append(result)

        print(f"Done ({elapsed:.1f}s)")
        print(
            f"  Accuracy: {val_accuracy:.4f} | F1: {val_f1:.4f} | AUC-ROC: {val_auc:.4f}"
        )

    # Select best model
    best_result = max(tuning_results, key=lambda x: x["auc_roc"])
    best_C = best_result["C"]
    best_pipeline = best_result["pipeline"]

    print("\n" + "-" * 50)
    print(f"Best C: {best_C} (AUC-ROC: {best_result['auc_roc']:.4f})")

    # -------------------------------------------------------------------------
    # 5. Final Training and Evaluation
    # -------------------------------------------------------------------------
    print("\n[5/6] Final training on train+validation...")

    # Combine train and validation
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)

    print(f"Final training set: {len(X_train_final):,} samples")

    # Create final pipeline
    final_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, max_categories=100
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    final_pipeline = Pipeline(
        [
            ("preprocessor", final_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=best_C,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    start_time = time.time()
    final_pipeline.fit(X_train_final, y_train_final)
    train_time = time.time() - start_time
    print(f"Training complete ({train_time:.1f}s)")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = final_pipeline.predict(X_test)
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Calculate all metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision_macro = precision_score(y_test, y_test_pred, average="macro")
    test_recall_macro = recall_score(y_test, y_test_pred, average="macro")
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_auc_roc = roc_auc_score(y_test, y_test_proba)

    # Per-class metrics
    test_precision_per_class = precision_score(y_test, y_test_pred, average=None)
    test_recall_per_class = recall_score(y_test, y_test_pred, average=None)
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision_macro:.4f} (macro)")
    print(f"Recall:    {test_recall_macro:.4f} (macro)")
    print(f"F1-Score:  {test_f1_macro:.4f} (macro)")
    print(f"AUC-ROC:   {test_auc_roc:.4f}")

    print("\nPer-Class Metrics:")
    print(f"  Class 0 (Not Purchased):")
    print(f"    Precision: {test_precision_per_class[0]:.4f}")
    print(f"    Recall:    {test_recall_per_class[0]:.4f}")
    print(f"    F1-Score:  {test_f1_per_class[0]:.4f}")
    print(f"  Class 1 (Purchased):")
    print(f"    Precision: {test_precision_per_class[1]:.4f}")
    print(f"    Recall:    {test_recall_per_class[1]:.4f}")
    print(f"    F1-Score:  {test_f1_per_class[1]:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(f"  [[TN={cm[0, 0]:,}  FP={cm[0, 1]:,}]")
    print(f"   [FN={cm[1, 0]:,}  TP={cm[1, 1]:,}]]")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_test_pred, target_names=["Not Purchased", "Purchased"]
        )
    )

    # Get validation metrics for the best model
    y_val_pred_best = best_pipeline.predict(X_val)
    y_val_proba_best = best_pipeline.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(y_val, y_val_pred_best)
    val_precision_macro = precision_score(y_val, y_val_pred_best, average="macro")
    val_recall_macro = recall_score(y_val, y_val_pred_best, average="macro")
    val_f1_macro = f1_score(y_val, y_val_pred_best, average="macro")
    val_auc_roc = roc_auc_score(y_val, y_val_proba_best)

    val_precision_per_class = precision_score(y_val, y_val_pred_best, average=None)
    val_recall_per_class = recall_score(y_val, y_val_pred_best, average=None)
    val_f1_per_class = f1_score(y_val, y_val_pred_best, average=None)

    # -------------------------------------------------------------------------
    # 6. Save Metrics to JSON
    # -------------------------------------------------------------------------
    print("\n[6/6] Saving metrics to JSON...")

    metrics = {
        "model": "LogisticRegression",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "best_C": best_C,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "data_split": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "train_val_size": int(len(X_train_final)),
            "total_size": int(len(X)),
        },
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "preprocessing": {
                "numerical": "StandardScaler",
                "categorical": "OneHotEncoder (max_categories=100)",
            },
        },
        "regularization_tuning": [
            {
                "C": r["C"],
                "val_accuracy": round(r["accuracy"], 4),
                "val_f1_macro": round(r["f1_macro"], 4),
                "val_auc_roc": round(r["auc_roc"], 4),
            }
            for r in tuning_results
        ],
        "validation_metrics": {
            "accuracy": round(val_accuracy, 4),
            "precision": {
                "macro": round(val_precision_macro, 4),
                "class_0": round(float(val_precision_per_class[0]), 4),
                "class_1": round(float(val_precision_per_class[1]), 4),
            },
            "recall": {
                "macro": round(val_recall_macro, 4),
                "class_0": round(float(val_recall_per_class[0]), 4),
                "class_1": round(float(val_recall_per_class[1]), 4),
            },
            "f1": {
                "macro": round(val_f1_macro, 4),
                "class_0": round(float(val_f1_per_class[0]), 4),
                "class_1": round(float(val_f1_per_class[1]), 4),
            },
            "auc_roc": round(val_auc_roc, 4),
        },
        "test_metrics": {
            "accuracy": round(test_accuracy, 4),
            "precision": {
                "macro": round(test_precision_macro, 4),
                "class_0": round(float(test_precision_per_class[0]), 4),
                "class_1": round(float(test_precision_per_class[1]), 4),
            },
            "recall": {
                "macro": round(test_recall_macro, 4),
                "class_0": round(float(test_recall_per_class[0]), 4),
                "class_1": round(float(test_recall_per_class[1]), 4),
            },
            "f1": {
                "macro": round(test_f1_macro, 4),
                "class_0": round(float(test_f1_per_class[0]), 4),
                "class_1": round(float(test_f1_per_class[1]), 4),
            },
            "auc_roc": round(test_auc_roc, 4),
        },
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    }

    metrics_path = script_dir / "../metrics/logistic_regression_metrics.json"
    metrics_path = metrics_path.resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
