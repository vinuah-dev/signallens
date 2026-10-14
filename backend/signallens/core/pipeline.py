"""The scikit-learn Pipeline: cast → impute → scale/encode → estimator.

Everything that learns from data (medians, modes, scaling, categories) is a
step *inside* the Pipeline, so ``fit`` on the training split is the only place
it learns — the test split and user inputs are only ever ``transform``-ed.
That is what keeps preprocessing from leaking test information."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Separator used in one-hot output names so they can be mapped back to the
# original column even when column names contain underscores.
NAME_SEP = "\x1f"
MISSING_CATEGORY = "(missing)"
_INDICATOR_PREFIX = "missingindicator_"


def combine_name(feature: str, category) -> str:
    return f"{feature}{NAME_SEP}{category}"


def source_feature(output_name: str) -> str:
    """Map a transformed column name back to the CSV column it came from."""
    if output_name.startswith(_INDICATOR_PREFIX):
        return output_name[len(_INDICATOR_PREFIX):]
    return output_name.split(NAME_SEP, 1)[0]


def to_category(value):
    """Canonical string form of a categorical value; 3, 3.0 and '3' are the
    same category. Missing stays NaN so the imputer can see it."""
    if value is None or value is pd.NA:
        return np.nan
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if float(value).is_integer():
            return str(int(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return str(value)


class ColumnCaster(BaseEstimator, TransformerMixin):
    """Coerce numeric features to float and categorical ones to canonical
    strings (keeping missing as NaN). Stateless, so it's safe to apply to the
    training split, test split and single prediction rows alike."""

    def __init__(self, numeric: list[str] | None = None, categorical: list[str] | None = None):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.numeric or []:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        for col in self.categorical or []:
            X[col] = X[col].map(to_category).astype(object)
        return X[(self.numeric or []) + (self.categorical or [])]

    def get_feature_names_out(self, input_features=None):
        return np.array((self.numeric or []) + (self.categorical or []), dtype=object)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    # Missingness is often informative (e.g. a field only filled in for one
    # class), so it is kept visible to the model: numeric gaps get the training
    # median *plus* a 0/1 "was missing" indicator, and categorical gaps become
    # their own "(missing)" category instead of being folded into the mode.
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=config.ONE_HOT_MAX_CATEGORIES,
                                sparse_output=False,
                                feature_name_combiner=combine_name,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(kinds: dict[str, str], estimator) -> Pipeline:
    numeric = [f for f, k in kinds.items() if k == "numeric"]
    categorical = [f for f, k in kinds.items() if k == "categorical"]
    return Pipeline(
        [
            ("cast", ColumnCaster(numeric, categorical)),
            ("prep", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ]
    )


def output_feature_names(pipeline: Pipeline) -> list[str]:
    return [str(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]
