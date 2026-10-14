"""HTTP contract: status codes, error shape and the full workflow."""

import pytest
from fastapi.testclient import TestClient

from signallens.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_algorithms_listed(client):
    keys = [a["key"] for a in client.get("/api/algorithms").json()]
    assert {"logistic_regression", "random_forest"} <= set(keys)


def test_upload_valid_csv(client, iris_bytes):
    r = client.post("/api/datasets", files={"file": ("iris.csv", iris_bytes, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["rows"] == 150
    assert body["profile"]["suggested_target"] == "species"
    assert len(body["preview"]["rows"]) == 50


def test_upload_wrong_format(client):
    r = client.post("/api/datasets", files={"file": ("data.xlsx", b"PK..", "application/octet-stream")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unsupported_format"


def test_upload_empty(client):
    r = client.post("/api/datasets", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "empty_file"


def test_unknown_ids_are_404(client):
    assert client.get("/api/datasets/nope").json()["error"]["code"] == "dataset_not_found"
    r = client.post("/api/models/nope/predict", json={"rows": [{}]})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_malformed_body_is_readable_422(client):
    r = client.post("/api/models", json={"dataset_id": "x"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "bad_request"


def test_full_workflow(client):
    ds = client.post("/api/samples/customer_churn").json()
    cols = {c["name"]: c for c in ds["profile"]["column_profiles"]}
    assert cols["customer_id"]["kind"] == "identifier"
    assert cols["signup_date"]["kind"] == "datetime"
    features = [n for n, c in cols.items() if c["kind"] in ("numeric", "categorical") and n not in ("churned", "cancellation_reason")]

    report = client.post(f"/api/datasets/{ds['id']}/frame", json={"target": "churned", "features": features + ["cancellation_reason"]}).json()
    assert any(i["code"] == "leakage_suspect" and i["column"] == "cancellation_reason" for i in report["issues"])

    bad = client.post(f"/api/datasets/{ds['id']}/frame", json={"target": "churned", "features": ["churned"]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "target_in_features"

    model = client.post(
        "/api/models",
        json={"dataset_id": ds["id"], "target": "churned", "features": features, "algorithm": "gradient_boosting", "balance": True},
    ).json()
    assert model["labels"] == ["No", "Yes"]
    assert model["evaluation"]["metrics"]["roc_auc"] > 0.7

    row = model["test_examples"]["rows"][0]["values"]
    pred = client.post(f"/api/models/{model['id']}/predict", json={"rows": [row, {**row, "tenure_months": 999}]}).json()
    assert len(pred["results"]) == 2
    assert any(n["code"].startswith("out_of_range") for n in pred["results"][1]["notes"])

    curve = client.post(f"/api/models/{model['id']}/what-if", json={"row": row, "feature": "contract"}).json()
    assert set(curve["x"]) == {"month-to-month", "one-year", "two-year"}

    invalid = client.post(f"/api/models/{model['id']}/predict", json={"rows": [{**row, "tenure_months": "ten"}]})
    assert invalid.status_code == 422 and "tenure_months" in invalid.json()["error"]["details"]["fields"]
