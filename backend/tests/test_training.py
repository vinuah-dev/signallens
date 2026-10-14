"""Model training and evaluation methodology."""

import numpy as np
import pandas as pd
import pytest

from signallens.core.algorithms import ALGORITHMS
from signallens.core.errors import FrameValidationError, TrainingError
from signallens.core.framing import FrameSpec
from signallens.core.training import TrainConfig, train_model

from .conftest import load_frame


def train(df, target="label", features=None, **cfg):
    data, profile = load_frame(df)
    features = features or [c for c in df.columns if c != target]
    return train_model(data, profile, "ds", FrameSpec(target, features), TrainConfig(**{"algorithm": "logistic_regression", **cfg}))


@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_every_algorithm_learns_a_separable_signal(separable_df, algorithm):
    model = train(separable_df, algorithm=algorithm)
    m = model.summary["evaluation"]["metrics"]
    assert m["accuracy"] > 0.9
    for key in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0.0 <= m[key] <= 1.0


def test_random_labels_score_near_baseline():
    """Sanity check against hard-coded/over-optimistic evaluation: with no
    signal the held-out score must not be meaningfully above chance."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600), "label": rng.choice(["x", "y"], 600)})
    m = train(df, algorithm="random_forest").summary["evaluation"]
    assert m["metrics"]["balanced_accuracy"] < 0.62


def test_split_is_stratified_and_sized(separable_df):
    ev = train(separable_df, test_size=0.25).summary["evaluation"]
    assert ev["n_test"] == 50 and ev["n_train"] == 150
    supports = {c["label"]: c["support"] for c in ev["per_class"]}
    assert supports == {"a": 25, "b": 25}


def test_confusion_matrix_is_consistent(separable_df):
    ev = train(separable_df).summary["evaluation"]
    matrix = np.array(ev["confusion"]["matrix"])
    assert ev["confusion"]["labels"] == ["a", "b"]
    assert matrix.sum() == ev["n_test"]
    assert np.trace(matrix) / matrix.sum() == pytest.approx(ev["metrics"]["accuracy"], abs=1e-4)
    assert matrix.sum(axis=1).tolist() == [c["support"] for c in ev["per_class"]]


def test_baseline_is_majority_class_rate():
    df = pd.DataFrame({"x": np.arange(100), "label": ["big"] * 80 + ["small"] * 20})
    ev = train(df).summary["evaluation"]
    assert ev["baseline"]["label"] == "big"
    assert ev["baseline"]["accuracy"] == pytest.approx(0.8)


def test_preprocessing_is_fitted_on_training_rows_only():
    """Leakage guard: imputation statistics must come from the training split,
    so test-set values can't influence how training data is transformed."""
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "label": rng.choice(["p", "q"], n)})
    df.loc[rng.choice(n, 20, replace=False), "x"] = np.nan
    model = train(df, test_size=0.3)
    prep = model.pipeline.named_steps["prep"]
    learned_median = prep.named_transformers_["num"].named_steps["impute"].statistics_[0]

    data, profile = load_frame(df)
    from sklearn.model_selection import train_test_split
    from signallens.core.framing import prepare_frame

    f = prepare_frame(data, profile, FrameSpec("label", ["x"]))
    X_train, _, _, _ = train_test_split(f.X, f.y, test_size=0.3, random_state=42, stratify=f.y)
    assert learned_median == pytest.approx(X_train["x"].median())
    assert learned_median != pytest.approx(f.X["x"].median())


def test_missingness_is_visible_to_the_model():
    """Regression (testing log #1): a field that is only filled for one class
    must stay distinguishable after imputation."""
    rng = np.random.default_rng(0)
    n = 400
    label = rng.choice(["churn", "stay"], n)
    reason = np.where(label == "churn", rng.choice(["price", "moved"], n), None)
    df = pd.DataFrame({"noise": rng.normal(size=n), "reason": reason, "label": label})
    ev = train(df).summary["evaluation"]
    assert ev["metrics"]["accuracy"] > 0.97


def test_training_is_reproducible(separable_df):
    a = train(separable_df, algorithm="random_forest").summary["evaluation"]["metrics"]
    b = train(separable_df, algorithm="random_forest").summary["evaluation"]["metrics"]
    assert a == b


def test_class_balancing_raises_minority_recall():
    rng = np.random.default_rng(11)
    n = 1000
    x = rng.normal(size=n)
    label = np.where(x + rng.normal(0, 1.2, n) > 1.6, "rare", "common")
    df = pd.DataFrame({"x": x, "label": label})
    recall = lambda m: next(c["recall"] for c in m.summary["evaluation"]["per_class"] if c["label"] == "rare")  # noqa: E731
    assert recall(train(df, balance=True)) > recall(train(df, balance=False))


def test_cross_validation_reported(separable_df):
    cv = train(separable_df).summary["cross_validation"]
    assert cv["folds"] == 5 and len(cv["scores"]) == 5
    assert all(0 <= s <= 1 for s in cv["scores"])


def test_importance_covers_every_feature_and_finds_signal(separable_df):
    imp = train(separable_df, algorithm="random_forest").summary["importance"]
    perm = imp["permutation"]["items"]
    assert {i["feature"] for i in perm} == {"signal", "noise", "colour"}
    assert perm[0]["feature"] == "signal"
    native = imp["native"]["items"]
    assert sum(i["share"] for i in native) == pytest.approx(1.0)
    assert {i["feature"] for i in native} == {"signal", "noise", "colour"}


def test_gradient_boosting_has_no_native_importance(separable_df):
    assert train(separable_df, algorithm="gradient_boosting").summary["importance"]["native"] is None


def test_interpretation_mentions_baseline(separable_df):
    lines = train(separable_df).summary["interpretation"]
    assert lines and "most common class" in lines[0]["text"]


def test_overfitting_is_called_out():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({f"f{i}": rng.normal(size=150) for i in range(8)} | {"label": rng.choice(["a", "b"], 150)})
    s = train(df, algorithm="decision_tree", params={"max_depth": None, "min_samples_leaf": 1}).summary
    assert any("overfitting" in line["text"] for line in s["interpretation"])


@pytest.mark.parametrize(
    "cfg, code",
    [
        ({"algorithm": "magic_forest"}, "unknown_algorithm"),
        ({"algorithm": "random_forest", "params": {"n_estimators": 5}}, "invalid_param"),
        ({"algorithm": "random_forest", "params": {"depth": 3}}, "unknown_param"),
        ({"algorithm": "logistic_regression", "params": {"C": "lots"}}, "invalid_param"),
        ({"algorithm": "logistic_regression", "test_size": 0.9}, "invalid_test_size"),
    ],
)
def test_invalid_configuration_rejected(separable_df, cfg, code):
    with pytest.raises(FrameValidationError) as exc:
        train(separable_df, **cfg)
    assert exc.value.code == code


def test_split_too_small_for_classes():
    df = pd.DataFrame({"x": range(12), "label": ["a", "b", "c", "d"] * 3})
    with pytest.raises(FrameValidationError) as exc:
        train(df, test_size=0.1)
    assert exc.value.code == "split_too_small"


def test_estimator_failure_becomes_training_error(separable_df, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def boom(self, *a, **k):
        raise ValueError("solver exploded")

    monkeypatch.setattr(LogisticRegression, "fit", boom)
    with pytest.raises(TrainingError) as exc:
        train(separable_df)
    assert "solver exploded" in exc.value.message


def test_imbalanced_headline_uses_balanced_accuracy():
    """Regression (testing log #5): with a dominant class the first finding must
    compare balanced accuracy, not plain accuracy, against the naive guess."""
    rng = np.random.default_rng(4)
    n = 800
    x = rng.normal(size=n)
    label = np.where(x + rng.normal(0, 0.6, n) > 1.3, "rare", "common")
    df = pd.DataFrame({"x": x, "label": label})
    summary = train(df, balance=True).summary
    assert summary["frame"]["imbalanced"]
    first = summary["interpretation"][0]["text"]
    assert "Balanced accuracy" in first and "never finding any other class" in first
    assert summary["interpretation"][0]["tone"] == "good"
