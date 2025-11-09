from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def main() -> None:
    df = pd.read_csv(Path("data/processed/diabetic_data_processed.csv"))
    target = "readmitted_30_days"
    X = df.drop(columns=[target])
    y = df[target]

    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
    thresholds = np.linspace(0.1, 0.95, 18)

    for scale in scales:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        scale_pos_weight=scale,
                        reg_lambda=1.0,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        scores = pipeline.predict_proba(X_test)[:, 1]
        print(f"=== scale_pos_weight={scale} ===")
        best = []
        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            positives = int(y_pred.sum())
            if precision >= 0.7:
                best.append((threshold, precision, recall, accuracy, positives))
        if best:
            for threshold, precision, recall, accuracy, positives in best:
                print(
                    f" threshold={threshold:.2f} -> precision={precision:.3f}, "
                    f"recall={recall:.3f}, accuracy={accuracy:.3f}, positives={positives}"
                )
        else:
            top = []
            for threshold in thresholds:
                y_pred = (scores >= threshold).astype(int)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred)
                positives = int(y_pred.sum())
                top.append((precision, threshold, recall, accuracy, positives))
            top.sort(reverse=True)
            print(" top precision settings:")
            for precision, threshold, recall, accuracy, positives in top[:5]:
                print(
                    f" threshold={threshold:.2f} -> precision={precision:.3f}, "
                    f"recall={recall:.3f}, accuracy={accuracy:.3f}, positives={positives}"
                )


if __name__ == "__main__":
    main()

