"""Prediction behaviour and input validation."""

import numpy as np
import pandas as pd
import pytest

from signallens.core.errors import PredictionInputError
from signallens.core.explain import what_if_curve
from signallens.core.framing import FrameSpec
from signallens.core.loading import load_csv
from signallens.core.prediction import predict_rows
from signallens.core.profiling import profile_dataset
from signallens.core.training import TrainConfig, train_model

from .conftest import load_frame


@pytest.fixture
def model(separable_df):
    data, profile = load_frame(separable_df)
    return train_model(data, profile, "ds", FrameSpec("label", ["signal", "noise", "colour"]), TrainConfig("logistic_regression"))


@pytest.fixture
def iris_model(iris_bytes):
    loaded = load_csv(iris_bytes, "iris.csv")
    profile = profile_dataset(loaded.df)
    features = [c for c in loaded.df.columns if c != "species"]
    return train_model(loaded.df, profile, "iris", FrameSpec("species", features), TrainConfig("random_forest"))


def one(model, row, explain=True):
    return predict_rows(model, [row], explain=explain)[0]


def test_prediction_shape(model):
    r = one(model, {"signal": 0.1, "noise": 0.0, "colour": "red"})
    assert r["predicted"] in ("a", "b")
    assert sum(p["probability"] for p in r["probabilities"]) == pytest.approx(1.0)
    assert r["confidence"] == max(p["probability"] for p in r["probabilities"])
    assert r["probabilities"][0]["label"] == r["predicted"]


def test_prediction_follows_the_learned_signal(model):
    assert one(model, {"signal": -1, "noise": 0, "colour": "red"})["predicted"] == "a"
    assert one(model, {"signal": 7, "noise": 0, "colour": "red"})["predicted"] == "b"


def test_predictions_match_the_pipeline(iris_model):
    rows = [ex["values"] for ex in iris_model.summary["test_examples"]["rows"][:10]]
    expected = iris_model.pipeline.predict(pd.DataFrame(rows))
    got = [r["predicted"] for r in predict_rows(iris_model, rows, explain=False)]
    assert got == list(expected)


def test_iris_extremes(iris_model):
    small = {"sepal_length_cm": 5.0, "sepal_width_cm": 3.4, "petal_length_cm": 1.4, "petal_width_cm": 0.2}
    large = {"sepal_length_cm": 7.2, "sepal_width_cm": 3.0, "petal_length_cm": 6.1, "petal_width_cm": 2.3}
    assert one(iris_model, small)["predicted"] == "setosa"
    assert one(iris_model, large)["predicted"] == "virginica"


def test_missing_values_are_imputed_with_a_note(model):
    r = one(model, {"signal": 6.5, "noise": None, "colour": ""})
    codes = {(n["feature"], n["code"]) for n in r["notes"]}
    assert ("noise", "imputed") in codes and ("colour", "imputed") in codes
    assert r["inputs"]["noise"] is None


def test_out_of_range_is_flagged_but_predicted(model):
    r = one(model, {"signal": 500, "noise": 0, "colour": "red"})
    assert any(n["code"] == "out_of_range_far" and n["feature"] == "signal" for n in r["notes"])
    assert r["predicted"] in ("a", "b")


def test_slightly_out_of_range(model):
    hi = next(s for s in model.schema if s["name"] == "signal")["max"]
    r = one(model, {"signal": hi + 0.5, "noise": 0, "colour": "red"})
    assert any(n["code"] == "out_of_range" for n in r["notes"])


def test_non_numeric_value_rejected(model):
    with pytest.raises(PredictionInputError) as exc:
        one(model, {"signal": "abc", "noise": 0, "colour": "red"})
    assert "signal" in exc.value.details["fields"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_non_finite_or_boolean_rejected(model, bad):
    with pytest.raises(PredictionInputError):
        one(model, {"signal": bad, "noise": 0, "colour": "red"})


def test_numeric_strings_accepted(model):
    r = one(model, {"signal": " 6.2 ", "noise": "0", "colour": "red"})
    assert r["inputs"]["signal"] == 6.2


def test_unseen_category_flagged(model):
    r = one(model, {"signal": 0, "noise": 0, "colour": "ultraviolet"})
    assert any(n["code"] == "unseen_category" for n in r["notes"])


def test_unknown_keys_ignored(model):
    r = one(model, {"signal": 0, "noise": 0, "colour": "red", "label": "b", "extra": 1})
    assert {n["feature"] for n in r["notes"] if n["code"] == "ignored"} == {"extra", "label"}


def test_batch_limits(model):
    with pytest.raises(PredictionInputError):
        predict_rows(model, [])
    with pytest.raises(PredictionInputError):
        predict_rows(model, [{"signal": 0}] * 101)


def test_sensitivity_points_at_the_signal(model):
    r = one(model, {"signal": 6.5, "noise": 0.0, "colour": "red"})
    sens = r["sensitivity"]
    assert {s["feature"] for s in sens} == {"signal", "noise", "colour"}
    assert sens[0]["feature"] == "signal" and sens[0]["delta"] > 0
    assert all(np.isfinite(s["delta"]) for s in sens)


def test_what_if_curve(iris_model):
    spec = next(s for s in iris_model.schema if s["name"] == "petal_length_cm")
    row = pd.DataFrame([{s["name"]: s["median"] for s in iris_model.schema}])
    curve = what_if_curve(iris_model.pipeline, row, spec)
    assert len(curve["x"]) > 10 and curve["x"] == sorted(curve["x"])
    totals = np.sum([s["values"] for s in curve["series"]], axis=0)
    assert np.allclose(totals, 1.0)
    setosa = next(s for s in curve["series"] if s["label"] == "setosa")["values"]
    assert setosa[0] > setosa[-1]  # short petals -> setosa, long petals -> not
