
"""Illustrative PD-to-score mapping. Not a regulated scorecard."""
import numpy as np
import pandas as pd

def pd_to_score(pd_values, base_score=600, base_odds=50,
                points_to_double_odds=20):
    """Higher score means lower PD, using a transparent odds mapping."""
    pd_values = np.clip(np.asarray(pd_values, dtype=float), 1e-6, 1 - 1e-6)
    odds = (1 - pd_values) / pd_values
    factor = points_to_double_odds / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return np.round(offset + factor * np.log(odds)).astype(int)

def assign_risk_band(pd_values):
    pd_values = np.asarray(pd_values, dtype=float)
    return pd.cut(
        pd_values,
        bins=[-np.inf, 0.05, 0.15, 0.30, np.inf],
        labels=["Low", "Moderate", "High", "Very High"]
    )

def score_applications(model, applications):
    probabilities = model.predict_proba(applications)[:, 1]
    return pd.DataFrame({
        "PD": probabilities,
        "Credit_Score": pd_to_score(probabilities),
        "Risk_Band": assign_risk_band(probabilities).astype(str)
    })
