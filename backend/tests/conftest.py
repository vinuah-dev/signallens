import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signallens.core.loading import load_csv  # noqa: E402
from signallens.core.profiling import profile_dataset  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "signallens" / "samples"


def to_csv_bytes(df: pd.DataFrame, sep: str = ",") -> bytes:
    return df.to_csv(index=False, sep=sep).encode("utf-8")


def load_frame(df: pd.DataFrame):
    loaded = load_csv(to_csv_bytes(df), "data.csv")
    return loaded.df, profile_dataset(loaded.df)


@pytest.fixture
def separable_df() -> pd.DataFrame:
    """Two well-separated classes plus a noise column and a categorical one."""
    rng = np.random.default_rng(0)
    n = 200
    label = np.array(["a", "b"] * (n // 2))
    signal = np.where(label == "a", rng.normal(0, 1, n), rng.normal(6, 1, n))
    return pd.DataFrame(
        {
            "signal": signal.round(3),
            "noise": rng.normal(0, 1, n).round(3),
            "colour": rng.choice(["red", "green", "blue"], n),
            "label": label,
        }
    )


@pytest.fixture
def iris_bytes() -> bytes:
    return (SAMPLES / "iris.csv").read_bytes()


@pytest.fixture
def churn_bytes() -> bytes:
    return (SAMPLES / "customer_churn.csv").read_bytes()
