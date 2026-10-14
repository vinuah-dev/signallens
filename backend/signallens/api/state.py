"""Application state: the dataset and model stores."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from ..core import config
from ..core.store import LRUStore
from ..core.training import TrainedModel


@dataclass
class DatasetRecord:
    id: str
    filename: str
    source: str  # "upload" | "sample"
    df: pd.DataFrame
    profile: dict
    warnings: list[dict]
    meta: dict
    created_at: float = field(default_factory=time.time)


datasets: LRUStore[DatasetRecord] = LRUStore(config.MAX_DATASETS, "dataset")
models: LRUStore[TrainedModel] = LRUStore(config.MAX_MODELS, "model")
