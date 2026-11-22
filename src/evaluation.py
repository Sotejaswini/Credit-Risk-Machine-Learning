
"""Discrimination, threshold, calibration, and explainability metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, brier_score_loss, average_precision_score
)

def ks_statistic(y_true, probabilities):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return float(np.max(tpr - fpr))

def calculate_gini(auc):
    return float(2 * auc - 1)

def evaluate_probabilities(y_true, probabilities, threshold=0.5):
    auc = roc_auc_score(y_true, probabilities)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(auc),
        "GINI": calculate_gini(auc),
        "KS": ks_statistic(y_true, probabilities),
        "Brier_Score": float(brier_score_loss(y_true, probabilities)),
        "Average_Precision": float(average_precision_score(y_true, probabilities)),
        "Threshold": float(threshold),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }

def decile_table(y_true, probabilities, n_bins=10):
    frame = pd.DataFrame({"actual": np.asarray(y_true),
                          "pd": np.asarray(probabilities)})
    frame["decile"] = pd.qcut(
        frame["pd"].rank(method="first"), q=n_bins,
        labels=False, duplicates="drop"
    ) + 1
    result = frame.groupby("decile").agg(
        customers=("actual", "size"),
        average_pd=("pd", "mean"),
        observed_default_rate=("actual", "mean"),
        defaults=("actual", "sum")
    ).reset_index()
    return result.sort_values("decile", ascending=False)
