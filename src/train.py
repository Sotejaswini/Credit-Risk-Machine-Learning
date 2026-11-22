
"""Train baseline Logistic Regression, Random Forest, and XGBoost models."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data_preprocessing import load_credit_data, split_features_target
from src.feature_engineering import engineer_features

RANDOM_STATE = 42

def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    except ImportError:
        print("xgboost is not installed; skipping XGBoost.")
    return models

def train_all(data_dir="data/raw", model_dir="models", report_dir="reports"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    df = load_credit_data(data_dir)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_features_target(df)
    results = []
    for name, estimator in get_models().items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        probabilities = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities))
        }
        results.append(metrics)
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipe, Path(model_dir) / f"{safe_name}.joblib")
        print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f}")
    import pandas as pd
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(Path(report_dir) / "training_model_comparison.csv", index=False)
    # Save a reproducible holdout set beside the raw data directory.
    processed_dir = Path(data_dir).resolve().parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    with open(processed_dir / "feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f, indent=2)
    return results_df

if __name__ == "__main__":
    print(train_all())
