
"""Credit-risk feature engineering for the UCI credit-card default dataset."""
import numpy as np
import pandas as pd

REPAYMENT_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

def engineer_features(data):
    df = data.copy()

    # The dataset's original codes 0, 1, 2, ... represent repayment status.
    # Preserve original columns and create interpretable summary features.
    rep_cols = [c for c in REPAYMENT_COLS if c in df.columns]
    if rep_cols:
        repayment = df[rep_cols].apply(pd.to_numeric, errors="coerce")
        df["MAX_REPAYMENT_STATUS"] = repayment.max(axis=1)
        df["AVG_REPAYMENT_STATUS"] = repayment.mean(axis=1)
        df["COUNT_DELAYED_MONTHS"] = (repayment > 0).sum(axis=1)
        df["ANY_SEVERE_DELAY"] = (repayment >= 2).any(axis=1).astype(int)

    bill_cols = [c for c in BILL_COLS if c in df.columns]
    pay_cols = [c for c in PAY_AMT_COLS if c in df.columns]

    if "LIMIT_BAL" in df.columns and bill_cols:
        total_bill = df[bill_cols].clip(lower=0).sum(axis=1)
        df["AVG_BILL_AMOUNT"] = df[bill_cols].mean(axis=1)
        df["MAX_BILL_AMOUNT"] = df[bill_cols].max(axis=1)
        df["CREDIT_UTILIZATION"] = (
            df["AVG_BILL_AMOUNT"] / df["LIMIT_BAL"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

    if pay_cols:
        df["AVG_PAYMENT_AMOUNT"] = df[pay_cols].mean(axis=1)
        df["TOTAL_PAYMENT_AMOUNT"] = df[pay_cols].sum(axis=1)

    if bill_cols and pay_cols:
        df["PAYMENT_TO_BILL_RATIO"] = (
            df[pay_cols].sum(axis=1) /
            df[bill_cols].abs().sum(axis=1).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

    # Avoid infinities; imputation is fitted inside sklearn pipelines.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
