"""Data validation: file format, shape and value cleaning."""

import numpy as np
import pandas as pd
import pytest

from signallens.core import config
from signallens.core.errors import DataValidationError
from signallens.core.loading import load_csv

from .conftest import to_csv_bytes

GOOD = b"a,b,label\n" + b"".join(f"{i},{i * 2},{'x' if i % 2 else 'y'}\n".encode() for i in range(20))


def expect_error(raw: bytes, filename: str, code: str):
    with pytest.raises(DataValidationError) as exc:
        load_csv(raw, filename)
    assert exc.value.code == code, exc.value.message
    return exc.value


def test_valid_csv_loads():
    loaded = load_csv(GOOD, "ok.csv")
    assert loaded.df.shape == (20, 3)
    assert loaded.delimiter == ","


@pytest.mark.parametrize("name", ["data.xlsx", "data.json", "data", "photo.png"])
def test_unsupported_extension_rejected(name):
    expect_error(GOOD, name, "unsupported_format")


def test_excel_gets_export_hint():
    err = expect_error(GOOD, "book.xlsx", "unsupported_format")
    assert "Save as" in err.message


def test_empty_file_rejected():
    expect_error(b"", "empty.csv", "empty_file")


def test_whitespace_only_rejected():
    expect_error(b"  \n\n  ", "blank.csv", "empty_file")


def test_header_only_rejected():
    expect_error(b"a,b,label\n", "header.csv", "no_rows")


def test_binary_disguised_as_csv_rejected():
    expect_error(b"PK\x03\x04\x00\x00binary", "sneaky.csv", "unsupported_format")


def test_single_column_rejected():
    raw = b"only\n" + b"".join(f"{i}\n".encode() for i in range(20))
    expect_error(raw, "one.csv", "too_few_columns")


def test_too_few_rows_rejected():
    expect_error(b"a,label\n1,x\n2,y\n", "tiny.csv", "too_few_rows")


def test_malformed_rows_rejected():
    raw = GOOD + b"1,2,3,4,5\n"
    expect_error(raw, "bad.csv", "malformed_csv")


def test_oversized_file_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    expect_error(GOOD, "big.csv", "file_too_large")


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_other_delimiters_detected(sep):
    df = pd.DataFrame({"x": range(15), "y": [i * 0.5 for i in range(15)], "label": ["p", "q", "r"] * 5})
    loaded = load_csv(to_csv_bytes(df, sep), "d.csv")
    assert loaded.delimiter == sep
    assert list(loaded.df.columns) == ["x", "y", "label"]


def test_question_mark_and_blank_are_missing():
    raw = b"a,b,label\n" + b"".join(f"{'?' if i == 3 else i},{'' if i == 4 else i},x\n".encode() for i in range(12))
    df = load_csv(raw, "m.csv").df
    assert df["a"].isna().sum() == 1
    assert df["b"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(df["a"])


def test_none_is_a_real_category():
    """Regression (testing log #3): 'none' must not be treated as missing."""
    raw = b"service,label\n" + b"".join(f"{'none' if i % 3 == 0 else 'fiber'},{i % 2}\n".encode() for i in range(15))
    df = load_csv(raw, "n.csv").df
    assert df["service"].isna().sum() == 0
    assert (df["service"] == "none").sum() == 5


def test_numeric_with_junk_tokens_coerced_with_warning():
    values = [str(i) for i in range(40)] + ["unknown"]
    raw = ("amount,label\n" + "".join(f"{v},{i % 2}\n" for i, v in enumerate(values))).encode()
    loaded = load_csv(raw, "j.csv")
    assert pd.api.types.is_numeric_dtype(loaded.df["amount"])
    assert loaded.df["amount"].isna().sum() == 1
    assert any(w["code"] == "coerced_values" and w["column"] == "amount" for w in loaded.warnings)


def test_currency_and_thousands_parsed():
    raw = b'price,label\n' + b"".join(f'"${1000 + i:,}",{i % 2}\n'.encode() for i in range(12))
    df = load_csv(raw, "c.csv").df
    assert df["price"].tolist()[:2] == [1000.0, 1001.0]


def test_mostly_text_column_is_not_coerced():
    raw = ("code,label\n" + "".join(f"{'A' if i % 2 else i},{i % 2}\n" for i in range(20))).encode()
    df = load_csv(raw, "t.csv").df
    assert df["code"].dtype == object


def test_duplicate_columns_renamed():
    raw = b"a,a,label\n" + b"".join(f"{i},{i},x\n".encode() for i in range(12))
    loaded = load_csv(raw, "dup.csv")
    assert list(loaded.df.columns) == ["a", "a_1", "label"]
    assert any(w["code"] == "duplicate_column" for w in loaded.warnings)


def test_empty_rows_and_columns_dropped():
    raw = b"a,empty,label\n" + b"".join(f"{i},,x\n".encode() for i in range(12)) + b",,\n,,\n"
    loaded = load_csv(raw, "e.csv")
    assert "empty" not in loaded.df.columns
    assert len(loaded.df) == 12


def test_whitespace_trimmed_from_cells_and_headers():
    raw = b" a , label \n" + b"".join(f"{i},  yes  \n".encode() for i in range(12))
    df = load_csv(raw, "w.csv").df
    assert list(df.columns) == ["a", "label"]
    assert set(df["label"]) == {"yes"}


def test_latin1_file_decodes():
    raw = "city,label\n".encode() + "".join(f"Zürich,{i % 2}\n" for i in range(12)).encode("latin-1")
    df = load_csv(raw, "l.csv").df
    assert df["city"].iloc[0] == "Zürich"


def test_large_file_is_sampled(monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 50)
    df = pd.DataFrame({"x": np.arange(200), "label": ["a", "b"] * 100})
    loaded = load_csv(to_csv_bytes(df), "big.csv")
    assert len(loaded.df) == 50
    assert any(w["code"] == "rows_sampled" for w in loaded.warnings)
