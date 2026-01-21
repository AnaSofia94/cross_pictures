from fastapi.testclient import TestClient

from cross_pictures.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"Status": "ok",
        "model_loaded": True}