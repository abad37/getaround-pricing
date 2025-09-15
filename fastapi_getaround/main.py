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
model_data = None  # attendu: dict {"pipeline","model_name","metrics","categorical_columns","numeric_columns","compat_feature_order"}

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
# Utils - Pricing
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

    # colonnes bool connues (si encodées comme numériques)
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
# Delay Analysis - chargement & helpers
# =========================
DELAY_FILE = os.getenv("DELAY_FILE", os.path.join(BASE_DIR, "get_around_delay_analysis.xlsx"))
_delay_df = None
_delay_error = None

def _load_delay_df():
    global _delay_df, _delay_error
    try:
        if not os.path.exists(DELAY_FILE):
            _delay_error = f"Fichier non trouvé: {DELAY_FILE}"
            _delay_df = None
            return
        df = pd.read_excel(DELAY_FILE)  # nécessite openpyxl dans requirements
        # Nettoyage minimal & features
        df["delay_at_checkout_in_minutes"] = df["delay_at_checkout_in_minutes"].fillna(0)
        df["time_delta_with_previous_rental_in_minutes"] = df["time_delta_with_previous_rental_in_minutes"].fillna(np.inf)
        df["is_late"] = df["delay_at_checkout_in_minutes"] > 0
        df["is_conflict_real"] = df["delay_at_checkout_in_minutes"] > df["time_delta_with_previous_rental_in_minutes"]
        df["checkin_type"] = df["checkin_type"].astype(str).str.lower()
        _delay_df = df
        _delay_error = None
        print(f"✅ Delay dataset chargé ({len(df)} lignes)")
    except Exception as e:
        _delay_df = None
        _delay_error = f"{type(e).__name__}: {e}"
        print("❌ Delay dataset error:", _delay_error)

# charge au démarrage
_load_delay_df()

def _subset_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope and scope.lower() == "connect":
        return df[df["checkin_type"].eq("connect")]
    return df

def _kpis_baseline(d: pd.DataFrame) -> dict:
    total = len(d)
    return {
        "locations_total": int(total),
        "retard_%": round(100 * d["is_late"].mean(), 2) if total else 0.0,
        "conflits_réels_%": round(100 * d["is_conflict_real"].mean(), 2) if total else 0.0,
        "delta_median_avec_prev_min": float(
            d["time_delta_with_previous_rental_in_minutes"].replace(np.inf, np.nan).median()
        ) if total else None,
        "retard_median_min": float(d["delay_at_checkout_in_minutes"].median()) if total else None,
    }

def _tradeoff(d: pd.DataFrame, thresholds: List[int]) -> List[dict]:
    out = []
    total = len(d)
    conflicts = int(d["is_conflict_real"].sum())
    for T in thresholds:
        affected = int((d["time_delta_with_previous_rental_in_minutes"] < T).sum())
        solved = int(((d["is_conflict_real"]) & (d["time_delta_with_previous_rental_in_minutes"] < T)).sum())
        out.append({
            "threshold_min": int(T),
            "offres_masquées": affected,
            "offres_masquées_%": round(100 * affected / total, 2) if total else 0.0,
            "conflits_réels": conflicts,
            "conflits_réels_%": round(100 * conflicts / total, 2) if total else 0.0,
            "conflits_résolus": solved,
            "conflits_résolus_%_des_conflits": round(100 * solved / (conflicts if conflicts else 1), 2)
        })
    return out

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
    <li><a href="/delay/health">/delay/health</a> — Dataset retards</li>
    <li><a href="/delay/kpis">/delay/kpis</a> — KPI retards (params: scope)</li>
    <li><a href="/delay/tradeoff">/delay/tradeoff</a> — Trade-off (params: scope, thresholds)</li>
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

# -------- Delay Analysis endpoints --------
@app.get("/delay/health")
def delay_health():
    ok = _delay_df is not None and _delay_error is None
    return {
        "available": ok,
        "rows": int(len(_delay_df)) if ok else 0,
        "file": DELAY_FILE,
        "error": _delay_error
    }

@app.get("/delay/kpis")
def delay_kpis(scope: str = "all"):
    if _delay_df is None:
        raise HTTPException(status_code=503, detail=_delay_error or "Dataset delay non disponible")
    d = _subset_scope(_delay_df, scope)
    return {"scope": scope, "kpis": _kpis_baseline(d)}

@app.get("/delay/tradeoff")
def delay_tradeoff(scope: str = "all", thresholds: str = "0,15,30,45,60,90,120,180,240"):
    if _delay_df is None:
        raise HTTPException(status_code=503, detail=_delay_error or "Dataset delay non disponible")
    try:
        th = [int(x) for x in thresholds.split(",") if str(x).strip() != ""]
    except Exception:
        raise HTTPException(status_code=400, detail="Param 'thresholds' invalide, ex: 0,30,60,90")
    d = _subset_scope(_delay_df, scope)
    table = _tradeoff(d, th)
    # petite reco simple (ex: 60 min si bon compromis)
    reco = None
    try:
        t60 = next((r for r in table if r["threshold_min"] == 60), None)
        if t60 and t60["conflits_résolus_%_des_conflits"] >= 60 and t60["offres_masquées_%"] <= 10:
            reco = f"Reco: seuil 60 min (bon compromis) sur scope '{scope}'"
    except Exception:
        pass
    return {"scope": scope, "thresholds": th, "tradeoff": table, "recommendation": reco}

@app.exception_handler(404)
async def not_found(req: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint non trouvé", "docs": f"{API_BASE_URL}/docs"}
    )

    # -------- Delay Analysis extra endpoints --------

@app.get("/delay/distribution")
def delay_distribution(bins: int = 40, scope: str = "all"):
    """
    Renvoie un histogramme (bins) sur delay_at_checkout_in_minutes + quelques stats.
    """
    if _delay_df is None:
        raise HTTPException(status_code=503, detail=_delay_error or "Dataset delay non disponible")
    d = _subset_scope(_delay_df, scope)
    series = d["delay_at_checkout_in_minutes"].clip(lower=0)  # on coupe les négatifs pour l'histo
    counts, edges = np.histogram(series, bins=int(bins))
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    return {
        "scope": scope,
        "bins": int(bins),
        "centers": [float(x) for x in centers],
        "counts": [int(x) for x in counts.tolist()],
        "late_rate_%": round(100 * d["is_late"].mean(), 2),
        "median_delay_min": float(series.median()) if len(series) else 0.0,
        "p95_delay_min": float(series.quantile(0.95)) if len(series) else 0.0
    }

@app.get("/delay/by-scope")
def delay_by_scope():
    """
    Compare KPIs all vs connect (retard et conflits réels).
    """
    if _delay_df is None:
        raise HTTPException(status_code=503, detail=_delay_error or "Dataset delay non disponible")
    def _k(d):
        return {
            "retard_%": round(100 * d["is_late"].mean(), 2),
            "conflits_réels_%": round(100 * d["is_conflict_real"].mean(), 2),
        }
    return {
        "all": _k(_delay_df),
        "connect": _k(_delay_df[_delay_df["checkin_type"].eq("connect")]),
    }

@app.get("/delay/revenue")
def delay_revenue(scope: str = "all", threshold: int = 60,
                  avg_price_day: float = 50.0, avg_duration_hours: float = 24.0):
    """
    Estimation simple du CA potentiel perdu par masquage des offres < threshold.
    Hypothèses: prix moyen par jour & durée moyenne (heures).
    """
    if _delay_df is None:
        raise HTTPException(status_code=503, detail=_delay_error or "Dataset delay non disponible")
    d = _subset_scope(_delay_df, scope)
    total = len(d)
    if total == 0:
        return {"scope": scope, "threshold": threshold, "revenue_lost_estimate": 0.0, "offres_masquées": 0}

    masked = int((d["time_delta_with_previous_rental_in_minutes"] < threshold).sum())
    # conversion simple: prix horaire moyen
    price_per_hour = float(avg_price_day) / 24.0
    estimated_hours = float(avg_duration_hours)
    revenue_lost = masked * price_per_hour * estimated_hours

    return {
        "scope": scope,
        "threshold": int(threshold),
        "offres_masquées": masked,
        "offres_masquées_%": round(100 * masked / total, 2),
        "avg_price_day": float(avg_price_day),
        "avg_duration_hours": float(avg_duration_hours),
        "revenue_lost_estimate": round(revenue_lost, 2)
    }
