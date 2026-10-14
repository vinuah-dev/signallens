"""Generate the bundled sample datasets.

    python scripts/make_samples.py

* iris.csv, breast_cancer.csv — classic public datasets shipped with scikit-learn.
* customer_churn.csv — SYNTHETIC. Generated below from a known formula so that
  it exercises the messy parts of the pipeline: an ID column, a date column,
  categorical features, missing values, junk tokens in a numeric column,
  class imbalance and one deliberately leaky column (cancellation_reason is only
  filled in for customers who churned)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

OUT = Path(__file__).resolve().parents[1] / "signallens" / "samples"


def iris() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = data.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length_cm",
            "sepal width (cm)": "sepal_width_cm",
            "petal length (cm)": "petal_length_cm",
            "petal width (cm)": "petal_width_cm",
        }
    )
    df["species"] = df.pop("target").map(dict(enumerate(data.target_names)))
    return df


def breast_cancer() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["diagnosis"] = df.pop("target").map({0: "malignant", 1: "benign"})
    return df


def customer_churn(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["card", "bank transfer", "electronic check", "mailed check"], n, p=[0.3, 0.25, 0.3, 0.15])
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.4, 0.15])
    senior = rng.choice(["Yes", "No"], n, p=[0.18, 0.82])
    tenure = np.clip(rng.gamma(2.0, 14, n), 1, 72).round().astype(int)
    monthly = np.clip(rng.normal(65, 22, n) + (internet == "fiber") * 18 - (internet == "none") * 30, 18, 130).round(2)
    tickets = rng.poisson(1.1 + (internet == "fiber") * 0.6, n)
    total = (monthly * tenure * rng.normal(1, 0.03, n)).round(2)
    signup = pd.Timestamp("2026-06-30") - pd.to_timedelta(tenure * 30, unit="D")

    logit = (
        -2.9
        + 2.2 * (contract == "month-to-month")
        - 1.6 * (contract == "two-year")
        + 0.9 * (payment == "electronic check")
        + 0.8 * (internet == "fiber")
        + 0.5 * (senior == "Yes")
        - 0.06 * tenure
        + 0.02 * (monthly - 65)
        + 0.45 * tickets
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))

    reasons = rng.choice(["price", "service quality", "moved", "competitor offer"], n)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "signup_date": signup.strftime("%Y-%m-%d"),
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "senior_citizen": senior,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total.astype(object),
            "support_tickets": tickets,
            "cancellation_reason": np.where(churned, reasons, "N/A"),
            "churned": np.where(churned, "Yes", "No"),
        }
    )
    # Realistic mess: gaps, a few junk tokens in a numeric column, unknown categories.
    for col, rate in [("internet_service", 0.04), ("monthly_charges", 0.05), ("support_tickets", 0.03), ("payment_method", 0.02)]:
        mask = rng.random(n) < rate
        df[col] = df[col].astype(object)
        df.loc[mask, col] = np.nan
    junk = rng.choice(n, 12, replace=False)
    df.loc[junk, "total_charges"] = "unknown"
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"iris": iris, "breast_cancer": breast_cancer, "customer_churn": customer_churn}.items():
        frame = fn()
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv  {frame.shape[0]} rows x {frame.shape[1]} cols")
