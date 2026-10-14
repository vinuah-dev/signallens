"""HTTP routes. Thin: parse the request, call the core, shape the response."""

from __future__ import annotations

import uuid
from functools import cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Query, UploadFile

from ..core import config
from ..core.algorithms import describe_algorithms
from ..core.errors import DataValidationError, NotFoundError, PredictionInputError
from ..core.explain import what_if_curve
from ..core.framing import FrameSpec, prepare_frame
from ..core.loading import load_csv
from ..core.prediction import predict_rows, validate_row
from ..core.profiling import preview_rows, profile_dataset
from ..core.training import TrainConfig, train_model
from . import state
from .schemas import FrameRequest, PredictRequest, TrainRequest, WhatIfRequest

router = APIRouter(prefix="/api")

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SAMPLES = {
    "iris": {
        "file": "iris.csv",
        "title": "Iris flowers",
        "blurb": "150 flowers, 4 measurements, 3 species. Clean and balanced — a good first run.",
        "tags": ["multiclass", "numeric", "clean"],
        "target": "species",
    },
    "breast_cancer": {
        "file": "breast_cancer.csv",
        "title": "Breast cancer diagnosis",
        "blurb": "569 tumour biopsies, 30 cell measurements, benign vs malignant.",
        "tags": ["binary", "numeric", "many features"],
        "target": "diagnosis",
    },
    "customer_churn": {
        "file": "customer_churn.csv",
        "title": "Customer churn (synthetic)",
        "blurb": "1,800 generated subscribers with gaps, categories, an ID, a date, imbalance and one leaky column.",
        "tags": ["binary", "mixed types", "missing values", "imbalanced"],
        "target": "churned",
    },
}


def _dataset_payload(rec: state.DatasetRecord) -> dict:
    return {
        "id": rec.id,
        "filename": rec.filename,
        "source": rec.source,
        "meta": rec.meta,
        "warnings": rec.warnings,
        "profile": rec.profile,
        "preview": preview_rows(rec.df, 0, 50),
    }


def _register(raw: bytes, filename: str, source: str) -> dict:
    loaded = load_csv(raw, filename)
    rec = state.DatasetRecord(
        id=uuid.uuid4().hex[:12],
        filename=filename,
        source=source,
        df=loaded.df,
        profile=profile_dataset(loaded.df),
        warnings=loaded.warnings,
        meta={"delimiter": loaded.delimiter, "encoding": loaded.encoding, "bytes": len(raw)},
    )
    state.datasets.put(rec.id, rec)
    return _dataset_payload(rec)


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "datasets": len(state.datasets), "models": len(state.models)}


@router.get("/algorithms")
def algorithms() -> list[dict]:
    return describe_algorithms()


@router.get("/limits")
def limits() -> dict:
    return {
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
        "max_rows": config.MAX_ROWS,
        "max_columns": config.MAX_COLUMNS,
        "min_rows": config.MIN_ROWS,
        "max_classes": config.MAX_CLASSES,
        "extensions": list(config.ALLOWED_EXTENSIONS),
        "test_size": {"min": config.MIN_TEST_SIZE, "max": config.MAX_TEST_SIZE, "default": config.DEFAULT_TEST_SIZE},
    }


@cache
def _sample_shape(filename: str) -> tuple[int, int]:
    df = pd.read_csv(SAMPLES_DIR / filename)
    return int(df.shape[0]), int(df.shape[1])


@router.get("/samples")
def samples() -> list[dict]:
    out = []
    for key, s in SAMPLES.items():
        path = SAMPLES_DIR / s["file"]
        if path.exists():
            rows, columns = _sample_shape(s["file"])
            meta = {k: v for k, v in s.items() if k != "file"}
            out.append({"key": key, **meta, "bytes": path.stat().st_size, "rows": rows, "columns": columns})
    return out


@router.post("/samples/{key}")
def load_sample(key: str) -> dict:
    if key not in SAMPLES:
        raise NotFoundError(f"No sample dataset called '{key}'.")
    path = SAMPLES_DIR / SAMPLES[key]["file"]
    return _register(path.read_bytes(), SAMPLES[key]["file"], "sample")


@router.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    # Read at most limit+1 bytes so an oversized upload is rejected without buffering it all.
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise DataValidationError(
            f"The file is larger than the {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.", code="file_too_large"
        )
    return _register(raw, file.filename or "", "upload")


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    return _dataset_payload(state.datasets.get(dataset_id))


@router.get("/datasets/{dataset_id}/rows")
def dataset_rows(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    missing_only: bool = False,
    sort_by: str | None = None,
    descending: bool = False,
) -> dict:
    rec = state.datasets.get(dataset_id)
    return preview_rows(rec.df, offset, limit, missing_only, sort_by, descending)


@router.post("/datasets/{dataset_id}/frame")
def check_frame(dataset_id: str, body: FrameRequest) -> dict:
    rec = state.datasets.get(dataset_id)
    frame = prepare_frame(rec.df, rec.profile, FrameSpec(body.target, body.features, body.categorical_overrides))
    return frame.report


@router.post("/models")
def train(body: TrainRequest) -> dict:
    rec = state.datasets.get(body.dataset_id)
    model = train_model(
        rec.df,
        rec.profile,
        rec.id,
        FrameSpec(body.target, body.features, body.categorical_overrides),
        TrainConfig(
            algorithm=body.algorithm,
            params=body.params,
            test_size=body.test_size,
            balance=body.balance,
            random_state=body.random_state,
            cross_validate=body.cross_validate,
        ),
    )
    state.models.put(model.id, model)
    return model.summary


@router.get("/models/{model_id}")
def get_model(model_id: str) -> dict:
    return state.models.get(model_id).summary


@router.post("/models/{model_id}/predict")
def predict(model_id: str, body: PredictRequest) -> dict:
    model = state.models.get(model_id)
    return {"model_id": model_id, "results": predict_rows(model, body.rows, explain=body.explain)}


@router.post("/models/{model_id}/what-if")
def what_if(model_id: str, body: WhatIfRequest) -> dict:
    model = state.models.get(model_id)
    spec = next((s for s in model.schema if s["name"] == body.feature), None)
    if spec is None:
        raise PredictionInputError(f"'{body.feature}' isn't a feature of this model.")
    clean, _ = validate_row(model.schema, body.row)
    row = pd.DataFrame([clean], columns=[s["name"] for s in model.schema])
    curve = what_if_curve(model.pipeline, row, spec)
    curve["current"] = clean[body.feature] if not (isinstance(clean[body.feature], float) and pd.isna(clean[body.feature])) else None
    return curve
