"""Turn raw uploaded bytes into a clean DataFrame, or explain why we can't.

Everything here is about the *file*: format, encoding, delimiter, shape.
Decisions about which columns are useful live in ``profiling``."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .errors import DataValidationError

_DELIMITERS = [",", ";", "\t", "|"]
_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_DECORATION = re.compile(r"[\s$€£₹%]")


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    filename: str
    delimiter: str
    encoding: str
    warnings: list[dict] = field(default_factory=list)


def _warn(warnings: list[dict], code: str, message: str, column: str | None = None) -> None:
    warnings.append({"level": "warn", "code": code, "message": message, "column": column})


def _check_filename(filename: str) -> None:
    name = (filename or "").lower().strip()
    if not name:
        raise DataValidationError("The upload has no file name.", code="no_file")
    if not name.endswith(config.ALLOWED_EXTENSIONS):
        ext = name.rsplit(".", 1)[-1] if "." in name else "(none)"
        hint = " Export it from Excel with 'Save as → CSV' first." if ext in ("xlsx", "xls", "ods") else ""
        raise DataValidationError(
            f"'.{ext}' files aren't supported — upload a CSV file.{hint}",
            code="unsupported_format",
        )


def _decode(raw: bytes) -> tuple[str, str]:
    if b"\x00" in raw[:8192]:
        raise DataValidationError(
            "This looks like a binary file (it contains null bytes), not a CSV text file.",
            code="unsupported_format",
        )
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DataValidationError("Couldn't decode the file as text.", code="unsupported_format")  # pragma: no cover


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS)).delimiter
    except csv.Error:
        pass
    # Fallback: the delimiter whose per-line count is most consistent and non-zero.
    lines = [line for line in sample.splitlines() if line.strip()][:20]
    best, best_score = ",", -1.0
    for d in _DELIMITERS:
        counts = [line.count(d) for line in lines]
        if not counts or min(counts) == 0:
            continue
        score = min(counts) - (max(counts) - min(counts))
        if score > best_score:
            best, best_score = d, score
    return best


def _dedupe_columns(columns: list[str], warnings: list[dict]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, col in enumerate(columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = name or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            new = f"{name}_{seen[name]}"
            _warn(warnings, "duplicate_column", f"Column name '{name}' appears more than once; renamed a copy to '{new}'.", new)
            name = new
        else:
            seen[name] = 0
        out.append(name)
    return out


def _clean_numeric_token(value: str) -> str:
    token = _DECORATION.sub("", value)
    return token.replace(",", "") if _THOUSANDS.match(token) else token


def _coerce_numeric_strings(df: pd.DataFrame, warnings: list[dict]) -> None:
    """Columns like '1,200' / '$5' / '42%' or numbers polluted by a few
    junk tokens become numeric; the junk becomes missing (and we say so)."""
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        present = s.dropna()
        if present.empty:
            continue
        numeric = pd.to_numeric(present.astype(str).map(_clean_numeric_token), errors="coerce")
        ok = int(numeric.notna().sum())
        if ok == 0 or ok / len(present) < 0.95:
            continue
        bad = present[numeric.isna()]
        df[col] = numeric.reindex(df.index)  # rows that were missing stay NaN
        if len(bad):
            examples = ", ".join(repr(v) for v in bad.astype(str).unique()[:3])
            _warn(
                warnings,
                "coerced_values",
                f"'{col}' is numeric but had {len(bad)} non-numeric value(s) ({examples}); treated them as missing.",
                col,
            )


def load_csv(raw: bytes, filename: str) -> LoadedDataset:
    _check_filename(filename)
    if raw is None or len(raw) == 0:
        raise DataValidationError("The file is empty (0 bytes).", code="empty_file")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise DataValidationError(f"The file is larger than the {mb} MB limit.", code="file_too_large")

    text, encoding = _decode(raw)
    if not text.strip():
        raise DataValidationError("The file only contains whitespace.", code="empty_file")

    delimiter = _sniff_delimiter(text)
    warnings: list[dict] = []
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            na_values=config.EXTRA_NA_VALUES,
            keep_default_na=True,
            skipinitialspace=True,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        raise DataValidationError("The file has no header row or data.", code="empty_file")
    except pd.errors.ParserError as exc:
        detail = str(exc).strip().splitlines()[-1]
        raise DataValidationError(
            f"The CSV is malformed: rows have inconsistent numbers of fields ({detail}).",
            code="malformed_csv",
        )

    # pandas silently renames duplicate headers to 'a.1'; re-read the raw header
    # so duplicates are renamed predictably *and* reported.
    header = next(csv.reader(io.StringIO(text), delimiter=delimiter), [])
    names = header if len(header) == df.shape[1] else list(df.columns)
    df.columns = _dedupe_columns(names, warnings)

    # Trim string cells; blank strings are missing values.
    for col in df.columns:
        if df[col].dtype == object:
            stripped = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = stripped.replace("", np.nan)
        elif df[col].dtype == bool:
            df[col] = df[col].map({True: "True", False: "False"})

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        _warn(warnings, "empty_columns_dropped", f"Dropped {len(empty_cols)} completely empty column(s): {', '.join(empty_cols[:5])}.")

    n_before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    if n_before - len(df):
        _warn(warnings, "empty_rows_dropped", f"Dropped {n_before - len(df)} completely empty row(s).")

    if len(df) == 0:
        raise DataValidationError("The file has a header but no data rows.", code="no_rows")
    if df.shape[1] < 2:
        raise DataValidationError(
            "Only one column was found. A classification dataset needs at least a target column and one feature"
            " — check that the file uses commas, semicolons, tabs or pipes as separators.",
            code="too_few_columns",
        )
    if df.shape[1] > config.MAX_COLUMNS:
        raise DataValidationError(
            f"The file has {df.shape[1]} columns; the limit is {config.MAX_COLUMNS}.", code="too_many_columns"
        )
    if len(df) < config.MIN_ROWS:
        raise DataValidationError(
            f"Only {len(df)} data row(s). At least {config.MIN_ROWS} are needed to train and test a model.",
            code="too_few_rows",
        )
    if len(df) > config.MAX_ROWS:
        _warn(
            warnings,
            "rows_sampled",
            f"The file has {len(df):,} rows; a random sample of {config.MAX_ROWS:,} is used to keep training responsive.",
        )
        df = df.sample(config.MAX_ROWS, random_state=0).sort_index().reset_index(drop=True)

    _coerce_numeric_strings(df, warnings)

    dupes = int(df.duplicated().sum())
    if dupes:
        _warn(
            warnings,
            "duplicate_rows",
            f"{dupes} row(s) are exact duplicates. They're kept, but duplicates can land in both the training and test"
            " split and make scores look better than they are.",
        )

    return LoadedDataset(df=df, filename=filename, delimiter=delimiter, encoding=encoding, warnings=warnings)
