from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from huggingface_hub import hf_hub_download

# =========================
# Config URLs (Spaces)
# =========================
GRADIO_DASH_URL = "https://adab82-gradio-getaround.hf.space"
API_BASE_URL    = "https://adab82-projet-getaround.hf.space"

# =========================
# Chargement modèle (LFS-safe)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOAD_ERROR = None
model_data = None  # attendu: dict {"pipeline", "model_name", "metrics", "categorical_columns","numeric_columns","compat_feature_order"}

def _is_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) < 1024:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(200).startswith("version https://git-lfs")
    except Exception:
        pass
    return False

try:
    model_path = LOCAL_MODEL_PATH
    if os.path.exists(LOCAL_MODEL_PATH) and _is_lfs_pointer(LOCAL_MODEL_PATH):
        # récupère le vrai binaire depuis ce Space (stocké en LFS)
        model_path = hf_hub_download("adab82/projet_getaround", repo_type="space", filename="model.pkl")
    elif not os.path.exists(LOCAL_MODEL_PATH):
        model_path = hf_hub_download("adab82/projet_getaround", repo_type="space", filename="model.pkl")

    model_data = joblib.load(model_path)
    print("✅ Modèle chargé depuis:", model_path)
except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}"
    print("❌ Impossible de charger le modèle:", LOAD_ERROR)
    model_data = None

# =========================
# FastAPI app + CORS
# =========================
app = FastAPI(
    title="GetAround Pricing API",
    description="API de prédiction de prix pour les locations de voitures GetAround",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (facultatif pour Gradio, inoffensif)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[GRADIO_DASH_URL, API_BASE_URL, "http://localhost:8501", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic models (v2 safe)
# =========================
class CarFeatures(BaseModel):
    model_key: str = Field(..., example="Citroën")
    mileage: int = Field(..., ge=0, le=500_000, example=50000)
    engine_power: int = Field(..., ge=40, le=500, example=120)
    fuel: str = Field(..., example="diesel")
    paint_color: str = Field(..., example="black")
    car_type: str = Field(..., example="hatchback")
    private_parking_available: bool = Field(..., example=True)
    has_gps: bool = Field(..., example=True)
    has_air_conditioning: bool = Field(..., example=True)
    automatic_car: bool = Field(..., example=False)
    has_getaround_connect: bool = Field(..., example=True)
    has_speed_regulator: bool = Field(..., example=True)
    winter_tires: bool = Field(..., example=False)
    model_config = {"protected_namespaces": ()}

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: str
    model_used: str
    timestamp: str
    model_config = {"protected_namespaces": ()}

class ModelInfo(BaseModel):
    model_name: str
    r2_score: float
    rmse: float
    mae: float
    features_count: int
    model_config = {"protected_namespaces": ()}

class CompatIn(BaseModel):
    input: List[List[float]]  # format exigé par l'énoncé
    model_config = {"protected_namespaces": ()}

class CompatOut(BaseModel):
    prediction: List[float]
    model_config = {"protected_namespaces": ()}

# =========================
# Utils
# =========================
def get_confidence_level(price: float) -> str:
    if price < 0:
        return "Faible"
    elif price < 50:
        return "Modéré"
    elif price < 200:
        return "Élevé"
    else:
        return "Très élevé"

def _coerce_types(v, target: str):
    # casting souple pour éviter les erreurs de type
    if target == "int":
        try:
            return int(v)
        except Exception:
            return None
    if target == "bool":
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return None
    # string
    return None if v is None else str(v)

def predict_with_pipeline_from_dict(payload: dict) -> float:
    """
    Inférence robuste: cast, valeurs par défaut, remplissage des manquants,
    alignement exact des colonnes brutes attendues par le pipeline.
    """
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modèle non dispo")
    pipeline = model_data.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline manquant dans le modèle")

    cat_cols = (model_data.get("categorical_columns") or [])
    num_cols = (model_data.get("numeric_columns") or [])
    cols = cat_cols + num_cols
    if not cols:
        raise HTTPException(status_code=500, detail="Colonnes attendues introuvables dans le modèle")

    # valeurs par défaut
    defaults_str = {c: "unknown" for c in cat_cols}
    defaults_num = {c: 0 for c in num_cols}

    # colonnes bool connues (si tu as encodé des bools comme numériques)
    bool_like = {
        "private_parking_available","has_gps","has_air_conditioning",
        "automatic_car","has_getaround_connect","has_speed_regulator","winter_tires"
    }

    row = {}
    for c in cat_cols:
        row[c] = _coerce_types(payload.get(c, defaults_str[c]), "str")

    for c in num_cols:
        if c in bool_like:
            row[c] = _coerce_types(payload.get(c, defaults_num[c]), "bool")
            row[c] = int(row[c]) if row[c] is not None else 0
        else:
            row[c] = _coerce_types(payload.get(c, defaults_num[c]), "int")
            row[c] = 0 if row[c] is None else row[c]

    X = pd.DataFrame([row], columns=cols).fillna(0)

    try:
        y = pipeline.predict(X)[0]
        return float(np.round(y, 2))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de prédiction (features manquantes ou types) : {e}. "
                   f"Attendu: cat={cat_cols}, num={num_cols}. Reçu: {row}"
        )

def predict_with_pipeline_from_matrix(matrix: List[List[float]]) -> List[float]:
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modèle non dispo")
    pipeline = model_data.get("pipeline")
    order = model_data.get("compat_feature_order")
    if pipeline is None or order is None:
        raise HTTPException(status_code=500, detail="Pipeline/feature order indisponible (retraine le modèle avec le bundle)")
    X = pd.DataFrame(matrix, columns=order)
    y = pipeline.predict(X)
    return [float(np.round(v, 2)) for v in y]

# =========================
# Endpoints
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8" /><title>GetAround Pricing API</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:24px}}a{{color:#0ea5e9}}</style>
</head><body>
  <h1>🚗 GetAround Pricing API</h1>
  <ul>
    <li><a href="/docs">/docs</a> — Swagger UI</li>
    <li><a href="/redoc">/redoc</a> — ReDoc</li>
    <li><a href="/model-info">/model-info</a> — Infos modèle</li>
    <li><a href="/health">/health</a> — État de l'API</li>
    <li><a href="/debug-config">/debug-config</a> — Configuration d'inférence</li>
  </ul>
  <p>Dashboard (Gradio) : <a href="{GRADIO_DASH_URL}" target="_blank">{GRADIO_DASH_URL}</a></p>
</body></html>
"""

@app.get("/health")
def health():
    return {"status": "healthy" if model_data else "unhealthy", "error": LOAD_ERROR}

@app.get("/debug-config")
def debug_config():
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modèle non dispo")
    return {
        "categorical_columns": model_data.get("categorical_columns"),
        "numeric_columns": model_data.get("numeric_columns"),
        "compat_feature_order": model_data.get("compat_feature_order"),
        "model_name": model_data.get("model_name"),
        "metrics": model_data.get("metrics"),
    }

@app.get("/model-info", response_model=ModelInfo)
def model_info():
    if not model_data:
        raise HTTPException(status_code=500, detail="Modèle non dispo")
    m = model_data.get("metrics", {}) or {}
    features_count = len((model_data.get("categorical_columns") or []) + (model_data.get("numeric_columns") or []))
    return ModelInfo(
        model_name=model_data.get("model_name", "Unknown"),
        r2_score=float(m.get("r2", 0.0)),
        rmse=float(m.get("rmse", 0.0)),
        mae=float(m.get("mae", 0.0)),
        features_count=features_count
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarFeatures):
    try:
        price = predict_with_pipeline_from_dict(car.model_dump())
        return PredictionResponse(
            predicted_price=price,
            confidence=get_confidence_level(price),
            model_used=model_data.get("model_name", "Unknown"),
            timestamp=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'entrée: {e}")

@app.post("/predict_compat", response_model=CompatOut)
def predict_compat(payload: CompatIn):
    preds = predict_with_pipeline_from_matrix(payload.input)
    return CompatOut(prediction=preds)

@app.exception_handler(404)
async def not_found(req: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint non trouvé", "docs": f"{API_BASE_URL}/docs"}
    )
