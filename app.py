"""
Banana Freshness API — FastAPI Backend (4 Classes)

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict   — upload image, get prediction
    GET  /health    — health check
    GET  /classes   — class info and day ranges
"""

import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from predict import load_model, predict

app = FastAPI(
    title="Banana Freshness API",
    description="Predict remaining shelf life of a banana from a photo.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/banana_model.pth")
model      = None
checkpoint = None

CLASS_LABELS = {
    'a': 'Green (unripe)',
    'b': 'Half-ripe',
    'c': 'Ripe',
    'd': 'Rotten',
}

CLASS_DESCRIPTIONS = {
    'a': '100% green skin, no yellow. Not ready to eat yet.',
    'b': 'Green-yellow mix, transitioning to ripe. Almost ready.',
    'c': 'Fully yellow with little or no spots. Perfect to eat now.',
    'd': 'Black or mushy skin. Do not consume.',
}

STORAGE_TIPS = {
    'a': 'Still unripe. Keep at room temperature away from sunlight. Do not refrigerate — it stops ripening. Will be ready in 5–8 days.',
    'b': 'Almost ripe! Leave at room temperature for 1–2 more days. Keep away from other fruits to control ripening speed.',
    'c': 'Perfectly ripe! Best time to eat fresh. Consume within 1–3 days. Refrigerate to slow further ripening.',
    'd': 'Rotten. Do not consume. Discard or compost immediately.',
}


@app.on_event("startup")
def load():
    global model, checkpoint
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}'. "
            "Train first: python train.py --data_root ./dataset"
        )
    model, checkpoint = load_model(MODEL_PATH)
    print(f"  Model loaded : {MODEL_PATH}")
    print(f"  Classes      : {checkpoint['classes']}")
    print(f"  Days map     : {checkpoint['days_map']}")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "classes":      checkpoint['classes'] if checkpoint else [],
    }


@app.get("/classes")
def classes():
    days_map = checkpoint['days_map'] if checkpoint else {}
    return {
        "classes": [
            {
                "id":          cls,
                "label":       CLASS_LABELS.get(cls, cls),
                "description": CLASS_DESCRIPTIONS.get(cls, ''),
                "days_min":    days_map.get(cls, {}).get('min', 0),
                "days_max":    days_map.get(cls, {}).get('max', 0),
                "storage_tip": STORAGE_TIPS.get(cls, ''),
            }
            for cls in (checkpoint['classes'] if checkpoint else [])
        ]
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    allowed = ['image/', 'application/octet-stream']
    if not any(file.content_type.startswith(t) for t in allowed):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        result = predict(img, model, checkpoint)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    cls      = result['class']
    days_est = result['days']

    return JSONResponse(content={
        "filename":      file.filename,
        "class":         cls,
        "label":         result['label'],
        "description":   CLASS_DESCRIPTIONS.get(cls, ''),
        "confidence":    round(result['confidence'] * 100, 1),
        "days": {
            "estimated": days_est,
            "min":       result['days_min'],
            "max":       result['days_max'],
            "message":   days_message(cls, days_est),
        },
        "storage_tip":   STORAGE_TIPS.get(cls, ''),
        "probabilities": {
            k: round(v * 100, 1)
            for k, v in result['probabilities'].items()
        },
    })


def days_message(cls: str, days: float) -> str:
    messages = {
        'a': f"Still unripe. Will be ready to eat in about {int(days)} days.",
        'b': f"Almost ripe! Will reach peak in about {int(days)} days.",
        'c': f"Perfectly ripe! Best consumed within {int(days)} day(s).",
        'd': "Rotten. Discard immediately.",
    }
    return messages.get(cls, f"Estimated {int(days)} days remaining.")


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
