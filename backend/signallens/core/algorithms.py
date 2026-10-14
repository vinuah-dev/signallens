"""Registry of supported classifiers and their tunable hyper-parameters.

The frontend renders its configuration form from ``describe_algorithms()``,
so adding a model here is the only change needed to expose it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .errors import FrameValidationError


@dataclass
class ParamSpec:
    name: str
    label: str
    type: str  # "int" | "float"
    default: Any
    min: float
    max: float
    help: str
    log: bool = False
    nullable: bool = False  # None = "no limit"
    step: float | None = None


@dataclass
class AlgorithmSpec:
    key: str
    label: str
    summary: str
    strengths: str
    native_importance: str | None  # "impurity" | "coefficients" | None
    params: list[ParamSpec] = field(default_factory=list)
    build: Callable[[dict, str | None, int], Any] = None  # type: ignore[assignment]


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "logistic_regression": AlgorithmSpec(
        key="logistic_regression",
        label="Logistic regression",
        summary="Draws straight-line boundaries between classes by weighting each feature.",
        strengths="Fast, stable and easy to read; struggles when relationships are curved or depend on combinations.",
        native_importance="coefficients",
        params=[
            ParamSpec("C", "Inverse regularisation (C)", "float", 1.0, 0.001, 100.0,
                      "Lower values force simpler weights (less overfitting); higher values fit the training data more closely.",
                      log=True),
        ],
        build=lambda p, cw, rs: LogisticRegression(C=p["C"], max_iter=3000, class_weight=cw, random_state=rs),
    ),
    "decision_tree": AlgorithmSpec(
        key="decision_tree",
        label="Decision tree",
        summary="Asks a sequence of yes/no questions about features, like a flowchart.",
        strengths="Captures thresholds and interactions; a single tree overfits easily if allowed to grow deep.",
        native_importance="impurity",
        params=[
            ParamSpec("max_depth", "Max depth", "int", 5, 1, 30, "How many questions deep the tree may go. Deeper = more detailed but more overfitting.", nullable=True, step=1),
            ParamSpec("min_samples_leaf", "Min rows per leaf", "int", 5, 1, 100, "Each final answer must be backed by at least this many training rows.", step=1),
        ],
        build=lambda p, cw, rs: DecisionTreeClassifier(
            max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], class_weight=cw, random_state=rs
        ),
    ),
    "random_forest": AlgorithmSpec(
        key="random_forest",
        label="Random forest",
        summary="Averages the votes of many decision trees, each trained on a random slice of the data.",
        strengths="Strong default for tabular data and resistant to overfitting; slower and harder to read than one tree.",
        native_importance="impurity",
        params=[
            ParamSpec("n_estimators", "Trees", "int", 200, 10, 600, "More trees give steadier votes but train more slowly.", step=10),
            ParamSpec("max_depth", "Max depth", "int", None, 1, 40, "Depth limit for each tree; empty = grow until leaves are pure.", nullable=True, step=1),
            ParamSpec("min_samples_leaf", "Min rows per leaf", "int", 1, 1, 50, "Larger values smooth out each tree's decisions.", step=1),
        ],
        build=lambda p, cw, rs: RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            class_weight=cw,
            random_state=rs,
            # n_jobs stays 1: parallel workers made the many small predict calls in
            # permutation importance ~3x slower than they saved during fit.
        ),
    ),
    "gradient_boosting": AlgorithmSpec(
        key="gradient_boosting",
        label="Gradient boosting",
        summary="Builds small trees one after another, each correcting the previous ones' mistakes.",
        strengths="Often the most accurate on tabular data; more sensitive to settings and easier to overfit on small data.",
        native_importance=None,
        params=[
            ParamSpec("learning_rate", "Learning rate", "float", 0.1, 0.01, 1.0, "How big a correction each new tree makes. Smaller = more careful, needs more rounds.", log=True),
            ParamSpec("max_iter", "Boosting rounds", "int", 200, 20, 600, "How many correcting trees to add.", step=10),
            ParamSpec("max_depth", "Max depth", "int", 3, 1, 20, "Depth of each small tree.", nullable=True, step=1),
        ],
        build=lambda p, cw, rs: HistGradientBoostingClassifier(
            learning_rate=p["learning_rate"],
            max_iter=p["max_iter"],
            max_depth=p["max_depth"],
            class_weight=cw,
            random_state=rs,
        ),
    ),
}


def describe_algorithms() -> list[dict]:
    return [
        {
            "key": a.key,
            "label": a.label,
            "summary": a.summary,
            "strengths": a.strengths,
            "native_importance": a.native_importance,
            "params": [p.__dict__ for p in a.params],
        }
        for a in ALGORITHMS.values()
    ]


def resolve_params(algorithm: str, params: dict | None) -> dict:
    """Fill defaults and validate user-supplied hyper-parameters."""
    if algorithm not in ALGORITHMS:
        raise FrameValidationError(
            f"Unknown algorithm '{algorithm}'. Choose one of: {', '.join(ALGORITHMS)}.", code="unknown_algorithm"
        )
    spec = ALGORITHMS[algorithm]
    params = dict(params or {})
    known = {p.name for p in spec.params}
    extra = set(params) - known
    if extra:
        raise FrameValidationError(f"Unknown setting(s) for {spec.label}: {', '.join(sorted(extra))}.", code="unknown_param")
    resolved = {}
    for p in spec.params:
        value = params.get(p.name, p.default)
        if value is None or value == "":
            if not p.nullable:
                raise FrameValidationError(f"{p.label} is required.", code="invalid_param")
            resolved[p.name] = None
            continue
        try:
            value = int(value) if p.type == "int" else float(value)
        except (TypeError, ValueError):
            raise FrameValidationError(f"{p.label} must be a number.", code="invalid_param")
        if not (p.min <= value <= p.max):
            raise FrameValidationError(f"{p.label} must be between {p.min:g} and {p.max:g}.", code="invalid_param")
        resolved[p.name] = value
    return resolved


def build_estimator(algorithm: str, params: dict, balance: bool, random_state: int):
    return ALGORITHMS[algorithm].build(params, "balanced" if balance else None, random_state)
