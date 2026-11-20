"""FastAPI server for the Food Photo to Recipe Finder application (R1, R5, R7)."""

from __future__ import annotations

import io
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

if TYPE_CHECKING:
    from backend.inference import AppInferenceService

_SERVICE: Optional["AppInferenceService"] = None
_SERVICE_LOCK = threading.Lock()
_LOAD_ERROR: Optional[str] = None
SAMPLES_DIR = Path("data/samples")
FRONTEND_DIST = Path("frontend/dist")

_SUGGESTED_QUERIES = {
    "pizza": "vegetarian under 30 minutes",
    "ramen": "quick spicy vegan noodle soup under 20 min",
    "pad_thai": "gluten free plant based under 20 minutes",
    "tacos": "spicy gluten free street tacos under 20 min",
    "sushi": "high protein spicy tuna or salmon under 20 min",
    "caesar_salad": "vegan chickpea tahini caesar in 15 min",
    "chocolate_cake": "gluten free dairy free chocolate cake",
    "hamburger": "vegan black bean burger under 25 min",
    "lasagna": "vegetarian vegetable lasagna",
    "pancakes": "quick vegan gluten free pancakes under 20 min",
}


def _get_service() -> "AppInferenceService":
    global _SERVICE, _LOAD_ERROR
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                try:
                    print("[Backend] Loading quantized ONNX & PyTorch models into memory...", flush=True)
                    from backend.inference import AppInferenceService

                    _SERVICE = AppInferenceService()
                    _LOAD_ERROR = None
                    print("[Backend] All models loaded and ready!", flush=True)
                except Exception as exc:
                    _LOAD_ERROR = str(exc)
                    print(f"[Backend] Model load error: {exc}", flush=True)
                    raise
    return _SERVICE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_get_service, daemon=True).start()
    yield


app = FastAPI(title="Food Photo to Recipe Finder API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": _SERVICE is not None, "load_error": _LOAD_ERROR}


@app.get("/api/samples")
def list_samples() -> dict:
    samples = []
    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.glob("*.jpg")):
            cls_name = p.stem
            samples.append({
                "name": p.name,
                "dish": cls_name,
                "display_name": cls_name.replace("_", " ").title(),
                "image_url": f"/api/samples/{p.name}",
                "suggested_query": _SUGGESTED_QUERIES.get(cls_name, "quick vegetarian under 25 minutes"),
            })
    return {"samples": samples}


@app.get("/api/samples/{filename}")
def get_sample_image(filename: str):
    target = (SAMPLES_DIR / filename).resolve()
    if not target.exists() or target.parent != SAMPLES_DIR.resolve():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(target)


@app.post("/api/predict")
async def predict(
    query: str = Form(""),
    compare: bool = Form(True),
    use_dish_hint: bool = Form(True),
    sample_name: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
) -> dict:
    service = _get_service()

    pil_img: Optional[Image.Image] = None
    if image is not None:
        raw = await image.read()
        try:
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc
    elif sample_name:
        target = (SAMPLES_DIR / sample_name).resolve()
        if not target.exists() or target.parent != SAMPLES_DIR.resolve():
            raise HTTPException(status_code=404, detail="Sample image not found")
        pil_img = Image.open(target).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="Provide either an uploaded image or sample_name")

    return service.predict(
        image=pil_img,
        query=query.strip(),
        compare=compare,
        use_dish_hint=use_dish_hint,
    )


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def root_redirect():
        return RedirectResponse(url="/docs")
