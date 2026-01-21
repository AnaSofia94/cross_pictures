import time
from fastapi import APIRouter, UploadFile, File, Request, HTTPException

from cross_pictures.preprocessing.image import preprocess_image
from cross_pictures.model.inference import run_inference
from cross_pictures.config import Settings
router = APIRouter()

@router.get("/health", include_in_schema=False)
async def health_check(request: Request):
    model_loaded = getattr(request.app.state, "model_loaded", True)
    return {
        "Status": "ok",
        "model_loaded": model_loaded,
    }


@router.post('/infer')
async def infer(request: Request, image: UploadFile | None = File(None), file: UploadFile = File(None)):
    if not request.app.state.model_loaded:
        raise HTTPException(status_code=503, detail = "Model not loaded")

    upload = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail = "Missing image file")

    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid Image File")

    imagebytes = await upload.read()
    input_tensor = preprocess_image(imagebytes)

    start = time.perf_counter()
    predictions = run_inference(
        session = request.app.state.session,
        input_tensor = input_tensor,
        labels = request.app.state.labels,
        top_k = int(Settings.TOP_K)
    )

    inference_times = (time.perf_counter() - start) * 1000

    return {
        "predictions": predictions,
        "inference_time_ms": round(inference_times,2)
    }


