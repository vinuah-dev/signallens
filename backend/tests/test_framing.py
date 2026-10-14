"""Target / feature selection validation."""

import numpy as np
import pandas as pd
import pytest

from signallens.core.errors import FrameValidationError
from signallens.core.framing import FrameSpec, prepare_frame

from .conftest import load_frame


def frame(df, target, features, overrides=None):
    data, profile = load_frame(df)
    return prepare_frame(data, profile, FrameSpec(target, features, overrides or []))


def expect(df, target, features, code):
    with pytest.raises(FrameValidationError) as exc:
        frame(df, target, features)
    assert exc.value.code == code, exc.value.message


def test_valid_frame(separable_df):
    f = frame(separable_df, "label", ["signal", "noise", "colour"])
    assert f.labels == ["a", "b"]
    assert f.kinds == {"signal": "numeric", "noise": "numeric", "colour": "categorical"}
    assert len(f.X) == len(f.y) == len(separable_df)


def test_missing_target_name(separable_df):
    expect(separable_df, "", ["signal"], "no_target")
    expect(separable_df, "nope", ["signal"], "target_not_found")


def test_no_features(separable_df):
    expect(separable_df, "label", [], "no_features")


def test_unknown_feature(separable_df):
    expect(separable_df, "label", ["signal", "ghost"], "feature_not_found")


def test_target_cannot_be_feature(separable_df):
    expect(separable_df, "label", ["signal", "label"], "target_in_features")


def test_single_class_target():
    df = pd.DataFrame({"x": range(20), "label": ["only"] * 20})
    expect(df, "label", ["x"], "single_class")


def test_continuous_target_rejected():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": range(50), "price": rng.normal(100, 20, 50).round(2)})
    expect(df, "price", ["x"], "target_continuous")


def test_identifier_target_rejected():
    df = pd.DataFrame({"x": range(40), "uid": [f"U{i}" for i in range(40)]})
    expect(df, "uid", ["x"], "target_not_categorical")


def test_too_many_classes():
    df = pd.DataFrame({"x": range(240), "cls": [f"c{i % 60}" for i in range(240)]})
    expect(df, "cls", ["x"], "too_many_classes")


def test_datetime_feature_rejected():
    df = pd.DataFrame(
        {"when": pd.date_range("2024-01-01", periods=30).strftime("%Y-%m-%d"), "x": range(30), "label": ["a", "b"] * 15}
    )
    expect(df, "label", ["when", "x"], "datetime_feature")


def test_rows_without_target_are_dropped(separable_df):
    df = separable_df.copy()
    df.loc[:4, "label"] = np.nan
    f = frame(df, "label", ["signal"])
    assert len(f.X) == len(df) - 5
    assert f.report["rows_dropped"] == 5
    assert any(i["code"] == "target_missing_rows" for i in f.report["issues"])


def test_singleton_class_dropped_with_warning(separable_df):
    df = separable_df.copy()
    df.loc[0, "label"] = "rare"
    f = frame(df, "label", ["signal"])
    assert "rare" not in f.labels
    assert any(i["code"] == "rare_classes_dropped" for i in f.report["issues"])


def test_integer_float_labels_normalised():
    df = pd.DataFrame({"x": range(30), "y": [0.0, 1.0, np.nan] * 10})
    f = frame(df, "y", ["x"])
    assert f.labels == ["0", "1"]


def test_imbalance_warning():
    df = pd.DataFrame({"x": range(100), "label": ["major"] * 90 + ["minor"] * 10})
    f = frame(df, "label", ["x"])
    assert f.report["imbalanced"]
    assert any(i["code"] == "class_imbalance" for i in f.report["issues"])


def test_leakage_suspect_flagged():
    rng = np.random.default_rng(0)
    label = rng.choice(["yes", "no"], 300)
    df = pd.DataFrame({"noise": rng.normal(size=300), "leak": np.where(label == "yes", "refund", "N/A"), "label": label})
    f = frame(df, "label", ["noise", "leak"])
    leaks = [i for i in f.report["issues"] if i["code"] == "leakage_suspect"]
    assert [i["column"] for i in leaks] == ["leak"]
    scores = {a["feature"]: a["score"] for a in f.report["association"]}
    assert scores["leak"] > 0.9 > scores["noise"]


def test_categorical_override(separable_df):
    df = separable_df.copy()
    df["rating"] = np.arange(len(df)) % 5 + 1
    f = frame(df, "label", ["signal", "rating"], overrides=["rating"])
    assert f.kinds["rating"] == "categorical"


def test_override_must_be_selected(separable_df):
    data, profile = load_frame(separable_df)
    with pytest.raises(FrameValidationError) as exc:
        prepare_frame(data, profile, FrameSpec("label", ["signal"], ["noise"]))
    assert exc.value.code == "bad_override"
