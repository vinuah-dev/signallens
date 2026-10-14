"""Request bodies. Responses are plain dicts built by the core."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core import config


class FrameRequest(BaseModel):
    target: str
    features: list[str]
    categorical_overrides: list[str] = Field(default_factory=list)


class TrainRequest(FrameRequest):
    dataset_id: str
    algorithm: str
    params: dict[str, Any] = Field(default_factory=dict)
    test_size: float = config.DEFAULT_TEST_SIZE
    balance: bool = False
    random_state: int = config.DEFAULT_RANDOM_STATE
    cross_validate: bool = True


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]
    explain: bool = True


class WhatIfRequest(BaseModel):
    row: dict[str, Any]
    feature: str
