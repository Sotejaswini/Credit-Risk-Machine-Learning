
"""Data loading, validation, and leakage-aware splitting utilities."""
from pathlib import Path
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "DEFAULT"
ID_COL = "ID"

def load_credit_data(data_dir="data/raw"):
    data_dir = Path(data_dir)
    csv_path = data_dir / "credit_risk_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        xls_path = data_dir / "default of credit card clients.xls"
        if not xls_path.exists():
            zip_path = data_dir / "default+of+credit+card+clients.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(data_dir)
            else:
                raise FileNotFoundError(
                    f"Dataset not found in {data_dir}. Place the UCI ZIP or Excel file there."
                )
        df = pd.read_excel(xls_path, sheet_name="Data", header=1)
    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(" ", "_", regex=False))
    if "default_payment_next_month" in df.columns:
        df = df.rename(columns={"default_payment_next_month": TARGET})
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("DEFAULT must contain only 0 and 1.")
    return df

def basic_quality_report(df):
    return {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_counts": df[TARGET].value_counts(dropna=False).to_dict(),
        "target_rate": float(df[TARGET].mean()),
    }

def split_features_target(df, test_size=0.20, random_state=42):
    """Stratified random split; this dataset has no reliable observation date."""
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
