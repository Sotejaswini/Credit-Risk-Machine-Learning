
"""Educational IFRS 9-style ECL calculations with explicit assumptions.

This module is a simplified learning example, not a production IFRS 9 engine.
A compliant implementation requires institution-specific policy, contractual
cash flows, forward-looking scenarios, discounting, staging governance,
lifetime term structures, validation, and accounting review.
"""
import numpy as np
import pandas as pd

def assign_stage(days_past_due, significant_increase=False,
                 credit_impaired=False):
    """Illustrative stage logic; actual SICR policy is entity-specific."""
    if credit_impaired or days_past_due >= 90:
        return 3
    if significant_increase or days_past_due >= 30:
        return 2
    return 1

def calculate_ecl(pd_value, lgd, ead, discount_factor=1.0,
                  scenario_weights=None, scenario_pd_multipliers=None):
    """Scenario-weighted ECL; inputs are fractions except EAD currency amount."""
    weights = scenario_weights or {"base": 1.0}
    multipliers = scenario_pd_multipliers or {k: 1.0 for k in weights}
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Scenario weights must sum to 1.")
    ecl = 0.0
    rows = []
    for scenario, weight in weights.items():
        scenario_pd = np.clip(pd_value * multipliers.get(scenario, 1.0), 0, 1)
        scenario_ecl = scenario_pd * lgd * ead * discount_factor
        ecl += weight * scenario_ecl
        rows.append({
            "scenario": scenario, "weight": weight,
            "scenario_pd": scenario_pd, "lgd": lgd,
            "ead": ead, "discount_factor": discount_factor,
            "scenario_ecl": scenario_ecl,
            "weighted_ecl": weight * scenario_ecl
        })
    return float(ecl), pd.DataFrame(rows)

def calculate_portfolio_ecl(df):
    required = {"PD", "LGD", "EAD", "STAGE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = df.copy()
    result["ECL"] = result["PD"] * result["LGD"] * result["EAD"]
    # Educational simplification: Stage 1 uses a 12-month PD supplied by user;
    # Stage 2/3 require lifetime PD / cash-shortfall modelling in real practice.
    return result
