import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost model for hospital readmission."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/diabetic_data_processed.csv"),
        help="Path to the processed CSV file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/readmission_xgb_pipeline.pkl"),
        help="Path to save the trained pipeline.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics.json"),
        help="Path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=Path("reports/confusion_matrix.png"),
        help="Path to save the confusion matrix plot.",
    )
    return parser.parse_args()


def build_pipeline(
    categorical_features: Iterable[str],
    numeric_features: Iterable[str],
    estimator,
) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    return pipeline


def evaluate_threshold(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    support = int(y_pred.sum())
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "predicted_positives": support,
    }


def generate_threshold_report(
    y_true: np.ndarray, y_scores: np.ndarray, thresholds: Iterable[float]
) -> List[Dict[str, float]]:
    return [evaluate_threshold(y_true, y_scores, thr) for thr in thresholds]


def choose_operating_point(
    rows: List[Dict[str, float]],
    min_precision: float,
    min_accuracy: float,
    min_recall: float,
) -> Dict[str, float]:
    candidates: List[Dict[str, float]] = [
        row
        for row in rows
        if row["precision"] >= min_precision
        and row["accuracy"] >= min_accuracy
        and row["recall"] >= min_recall
    ]
    if candidates:
        candidates.sort(
            key=lambda row: (row["precision"], row["recall"], -abs(row["threshold"] - 0.5)),
            reverse=True,
        )
        selected = candidates[0].copy()
        selected["meets_constraints"] = True
        return selected

    positive_recall_rows = [row for row in rows if row["recall"] > 0]
    ranking_key = lambda row: (
        row["recall"],
        row["precision"],
        row["accuracy"],
        -abs(row["threshold"] - 0.5),
    )
    fallback_base = (
        max(positive_recall_rows, key=ranking_key)
        if positive_recall_rows
        else max(rows, key=ranking_key)
    )
    fallback_copy = fallback_base.copy()
    fallback_copy["meets_constraints"] = False
    return fallback_copy


def build_model_specs(class_ratio: float) -> List[Tuple[str, object]]:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        scale_pos_weight=class_ratio,
        reg_lambda=1.0,
        min_child_weight=2,
    )

    log_reg = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    return [
        ("xgboost", xgb),
        ("logistic_regression", log_reg),
    ]


def plot_confusion_matrix(y_true, y_scores, threshold: float, output_path: Path) -> None:
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Readmission Within 30 Days - Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_path)
    target_column = "readmitted_30_days"
    feature_df = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_features = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = feature_df.select_dtypes(exclude=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    class_ratio = neg_count / pos_count if pos_count > 0 else 1.0

    thresholds = np.linspace(0.05, 0.95, 19)
    target_precision = 0.75
    target_accuracy = 0.7
    target_recall = 0.05

    threshold_records: List[Dict[str, float]] = []
    model_results: List[Dict[str, object]] = []

    for model_name, estimator in build_model_specs(class_ratio):
        pipeline = build_pipeline(categorical_features, numeric_features, estimator)
        pipeline.fit(X_train, y_train)

        if hasattr(pipeline, "predict_proba"):
            y_scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            # Fallback to decision_function and sigmoid transform
            decision_scores = pipeline.decision_function(X_test)
            y_scores = 1 / (1 + np.exp(-decision_scores))

        rows = generate_threshold_report(y_test.to_numpy(), y_scores, thresholds)
        for row in rows:
            record = row.copy()
            record["model"] = model_name
            threshold_records.append(record)

        selected_row = choose_operating_point(
            rows, target_precision, target_accuracy, target_recall
        )
        model_results.append(
            {
                "name": model_name,
                "pipeline": pipeline,
                "scores": y_scores,
                "selected": selected_row,
            }
        )

    viable_models = [
        result for result in model_results if result["selected"]["meets_constraints"]
    ]
    if viable_models:
        viable_models.sort(
            key=lambda result: (
                result["selected"]["precision"],
                result["selected"]["recall"],
                -abs(result["selected"]["threshold"] - 0.5),
            ),
            reverse=True,
        )
        best_result = viable_models[0]
    else:
        best_result = max(
            model_results,
            key=lambda result: (
                result["selected"]["precision"],
                result["selected"]["recall"],
                result["selected"]["accuracy"],
            ),
        )

    best_pipeline = best_result["pipeline"]
    y_scores_test = best_result["scores"]
    selected_row = best_result["selected"]
    threshold = float(selected_row["threshold"])
    y_pred_threshold = (y_scores_test >= threshold).astype(int)
    precision = float(selected_row["precision"])
    recall = float(selected_row["recall"])
    accuracy = float(selected_row["accuracy"])
    meets_constraints = bool(selected_row["meets_constraints"])
    predicted_positives = int(selected_row["predicted_positives"])

    metrics = {
        "threshold": float(threshold),
        "model": best_result["name"],
        "scale_pos_weight": float(class_ratio),
        "test_precision": precision,
        "test_recall": recall,
        "test_accuracy": accuracy,
        "predicted_positives": predicted_positives,
        "meets_constraints": meets_constraints,
        "target_precision": target_precision,
        "target_accuracy": target_accuracy,
        "target_recall": target_recall,
    }

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    threshold_report_path = Path("reports/threshold_metrics.csv")
    pd.DataFrame(threshold_records).to_csv(threshold_report_path, index=False)

    model_metrics_path = Path("reports/model_metrics.json")
    serialisable_model_metrics = [
        {
            "model": result["name"],
            "threshold": float(result["selected"]["threshold"]),
            "precision": float(result["selected"]["precision"]),
            "recall": float(result["selected"]["recall"]),
            "accuracy": float(result["selected"]["accuracy"]),
            "predicted_positives": int(result["selected"]["predicted_positives"]),
            "meets_constraints": bool(result["selected"]["meets_constraints"]),
        }
        for result in model_results
    ]
    with model_metrics_path.open("w") as f:
        json.dump(serialisable_model_metrics, f, indent=2)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_pipeline.threshold_ = float(threshold)
    joblib.dump(best_pipeline, args.model_path)

    plot_confusion_matrix(y_test, y_scores_test, threshold, args.confusion_matrix_path)
    print(f"Saved pipeline to {args.model_path}")
    print(f"Saved metrics to {args.metrics_path}")
    print(f"Saved confusion matrix to {args.confusion_matrix_path}")


if __name__ == "__main__":
    main()

