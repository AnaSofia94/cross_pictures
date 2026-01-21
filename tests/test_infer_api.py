from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from cross_pictures.main import app

client = TestClient(app)

def test_infer_endpoint_mocked():
    app.state.model_loaded = True
    app.state.session = object()
    app.state.labels = ["cat", "dog", "car"]

    fake_predictions = [
        {"label": "cat", "confidence": 0.9},
        {"label": "dog", "confidence": 0.1}
    ]

    with patch("cross_pictures.route.api.run_inference") as mock_run:
        mock_run.return_value = fake_predictions

        response = client.post("/infer",
                               files={"file": ("image.png", "image.png")
                                      },
                               )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "inference_time_ms" in data
        assert data["predictions"] == fake_predictions


