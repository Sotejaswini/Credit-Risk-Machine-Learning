
"""Population Stability Index (PSI) and feature drift monitoring."""
import numpy as np
import pandas as pd

def calculate_psi(expected, actual, bins=10, epsilon=1e-6):
    """PSI for numeric samples using expected-population quantile bins."""
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.empty or actual.empty:
        raise ValueError("Both expected and actual samples must contain values.")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        edges = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()), bins + 1
        )
    edges[0], edges[-1] = -np.inf, np.inf
    exp_counts = pd.cut(expected, bins=edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    act_counts = pd.cut(actual, bins=edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    exp_pct = np.clip(exp_counts.to_numpy(), epsilon, None)
    act_pct = np.clip(act_counts.to_numpy(), epsilon, None)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    detail = pd.DataFrame({
        "expected_pct": exp_pct,
        "actual_pct": act_pct,
        "psi_component": (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    })
    return float(psi), detail

def psi_interpretation(psi):
    if psi < 0.10:
        return "Low population shift (rule-of-thumb)"
    if psi < 0.25:
        return "Moderate population shift (investigate)"
    return "Large population shift (investigate urgently)"
