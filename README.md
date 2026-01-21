## Image Infernce Service

This project performs image Classification using a pre-trained ResNet-50 ONNX model. The service accepts an image via HTTP API, 
runs CPU-based inference and returns top-k predictions.

## Requirements

- Python 3.11+
- Poetry
- ONNX Runtime (CPU)

## Setup

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd cross_pictures
poetry install
```
Download the models and labels:

```bash
mkdir -p resources
curl -L -o resources/resnet50.onnx https://huggingface.co/onnxmodelzoo/legacy_models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx?raw=true
curl -L -o resources/imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

## Configuration

The service is configured via environment variables:

```bash
MODEL_PATH=resources/resnet50-v2-7.onnx
LABELS_PATH=resources/imagenet_classes.txt
TOP_k=3
NUM_THREADS=4
```

You can place these in a .env file or export them in the shell

## Running the service

Start the API using uvicorn:
```bash
poetry run uvicorn cross_pictures.main:app --reload
```

The service will be available at:
- http://127.0.0.1:8000

## API Endpoints
 
### Health Check
```md
### GET /health
Returns service health and model readiness.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Inference

```md
### POST /infer

Accepts an image file and returns the top-K predictions.

Example request:

curl -X POST http://127.0.0.1:8000/infer \
  -F "image=@cat.jpg"
```

Example response

```json
{
  "predictions": [
    { "label": "tiger cat", "confidence": 0.41 },
    { "label": "tabby", "confidence": 0.34 },
    { "label": "Egyptian cat", "confidence": 0.24 }
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