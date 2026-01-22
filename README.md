## Image Inference Service

Image classification service (FastAPI) using a pre-trained ResNet-50 ONNX model.
Send an image via HTTP, get top-k predictions back (CPU inference).

## Features

- FastAPI HTTP service
- ONNX Runtime (CPU inference)
- ImageNet preprocessing (224×224 + normalization)
- Top-K predictions with confidence scores
- Model loaded once at startup for performance
- Unit + API tests (mocked inference)
- Poetry-based dependency management


## Requirements
- Python 3.11.x (local)
- Poetry

## Quick start
```bash
git clone <repo-url>
cd cross_pictures
poetry install
poetry run uvicorn cross_pictures.main:app --reload
```

The bundled model (`resources/resnet50-v2-7.onnx`) and labels
(`resources/imagenet_classes.txt`) are already in the repo, so no extra download
is needed.

## Configuration
Environment variables (all have defaults):
```
MODEL_PATH=resources/resnet50-v2-7.onnx
LABELS_PATH=resources/imagenet_classes.txt
TOP_K=3
NUM_THREADS=4
```
You can export them or put them in a `.env`. Use absolute paths if you move the files.

## Project Structure
The project follows a layered architecture to keep responsibilities clearly separated
between the API layer, preprocessing logic, and model inference.  
This improves maintainability, testability, and performance.
```jsunicoderegexp
src/
cross_pictures/
main.py              # FastAPI app creation & startup lifecycle (model loading)
config.py            # Environment-based configuration
route/
  api.py             # HTTP endpoints (/health, /infer)

preprocessing/
  image.py           # Image resize, normalization, tensor conversion

model/
  loader.py          # ONNX model & labels loading
  inference.py       # Inference execution + top-k postprocessing
  
tests/
test_preprocessing.py  # Unit test for image preprocessing
test_infer_api.py      # Mocked API-level test for /infer

resources/
resnet50.onnx          # Pretrained ONNX model (included for convenience)
imagenet_classes.txt   # ImageNet label map
```

### Responsibilities

- **API layer** → Handles HTTP requests/responses and validation  
- **Preprocessing** → Transforms raw images into model-ready tensors  
- **Model layer** → Runs ONNX inference and post-processes predictions  
- **Startup lifecycle** → Loads the model once to avoid per-request overhead  
- **Tests** → Unit and API tests ensure correctness without depending on the real model  

This structure keeps components loosely coupled and easy to extend or test independently.


## Run
```bash
poetry run uvicorn cross_pictures.main:app --reload --host 0.0.0.0 --port 8000
```
App lives at http://127.0.0.1:8000

## API Endpoints
 
### Health Check
### GET /health
Returns service health and model readiness.
Example:
```
{
  "Status": "ok",
  "model_loaded": true
}
```

### Inference

```md
### POST /infer
Multipart form upload. Field name can be `file` or `image`.
Example:
```
curl -X POST http://127.0.0.1:8000/infer \
  -F "image=@cat.jpg"
```
Response (shape):
```
{
  "predictions": [
    {"label": "tiger cat", "probability": 0.41},
    {"label": "tabby", "probability": 0.34},
    {"label": "Egyptian cat", "probability": 0.24}
  ],
  "inference_time_ms": 31.2
}
```

## Testing

Run the test suite with:

```bash
poetry run pytest
```

## Design Decisions and Tradeoffs

- The ONNX model and labels are loaded once at application startup to avoid
  repeated initialization overhead and ensure consistent performance.
- Inference is executed synchronously, as ONNX Runtime is CPU-bound, while
  the API remains asynchronous.
- Image preprocessing is isolated from the API layer to improve testability
  and separation of concerns.
- Inference is mocked in API tests to keep tests fast and deterministic.

## Improvements with more time
- Adding basic metrics
- Supporting batch inference
- Providing a docker deployment