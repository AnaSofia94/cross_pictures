import os

from fastapi import FastAPI

from cross_pictures.model.loader import load_model, load_labels
from cross_pictures.route.api import router
from cross_pictures.config import Settings


app = FastAPI()

@app.on_event("startup")
async def startup():
    try:
        app.state.session = load_model(
            model_path=Settings.MODEL_PATH,
            num_threads=Settings.NUM_THREADS,
        )
        app.state.labels = load_labels(Settings.LABELS_PATH)
        app.state.model_loaded = True
    except Exception as e:
        app.state.model_loaded = False
        app.state.load_error = str(e)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "cross_pictures.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )





