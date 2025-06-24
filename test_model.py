import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_positive_sentiment():
    response = client.post(
        "/predict/trained",
        json={"text": "This is Amazing!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "LABEL_1"
    assert data["score"] > 0.5

def test_negative_sentiment():
    response = client.post(
        "/predict/trained",
        json = {"text": "This is Terrible!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "LABEL_0"
    assert data["score"] > 0.5