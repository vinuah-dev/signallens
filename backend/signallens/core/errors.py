"""Domain errors. Each carries a stable machine-readable ``code`` so the
frontend can choose a specific empty/error state, plus a human message."""

from __future__ import annotations


class SignalLensError(Exception):
    status = 400
    code = "error"

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details or {}


class DataValidationError(SignalLensError):
    """The uploaded file or its contents cannot be used."""

    status = 422
    code = "invalid_data"


class FrameValidationError(SignalLensError):
    """The chosen target/features cannot be used for classification."""

    status = 422
    code = "invalid_selection"


class TrainingError(SignalLensError):
    """The estimator failed to fit or evaluate."""

    status = 500
    code = "training_failed"


class PredictionInputError(SignalLensError):
    status = 422
    code = "invalid_input"


class NotFoundError(SignalLensError):
    status = 404
    code = "not_found"
