import glob
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import librosa
from typing import Literal, List
from starlette.status import HTTP_400_BAD_REQUEST
import uuid
import time
from sqlalchemy.orm import Session

# Importaciones locales
from auth import get_current_user_id
from database import SessionLocal
from models import Prediccion

# Configuración
MODEL_PATH = Path("./models/rf_cough_epoc_v1.pkl")
N_MFCC = 13
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# FastAPI
app = FastAPI(
    title="Modelo EPOC",
    description="Clasificador de tos entre pacientes con EPOC y sanos",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://id-preview--5341bf68-42ab-4af9-ba52-dd9e41fde1ae.lovable.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class PredictionResult(BaseModel):
    label: Literal["COPD", "Healthy"]
    probability: float
    filename: str
    execution_time: float

class PrediccionResponse(BaseModel):
    id: int
    nombre_archivo: str
    resultado: str
    user_id: int
    tiempo_ejecucion: float = Field(..., description="Execution time in seconds")
    fecha: str = Field(..., description="Prediction date in ISO format")

    class Config:
        from_attributes = True  # 👈 Esto es lo que necesitas


class UserPredictionsResponse(BaseModel):
    total: int = Field(..., description="Total number of predictions")
    predicciones: List[PrediccionResponse]
    resumen: dict

# Dependencias

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utilidades

def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Modelo no encontrado. Entrénalo primero.")
    model = joblib.load(MODEL_PATH)

@app.post("/predict", response_model=PredictionResult, tags=["predict"])
def predict(
        file: UploadFile = File(...),
        user_id: int = Depends(get_current_user_id),
        db: Session = Depends(get_db)
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida")

    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Solo se permiten archivos .wav")

    unique_name = f"{uuid.uuid4()}.wav"
    tmp_path = TEMP_DIR / unique_name

    with open(tmp_path, "wb") as f:
        f.write(file.file.read())

    start = time.perf_counter()

    try:
        feats = extract_features(str(tmp_path))
        proba = model.predict_proba(feats)[0]
        pred = model.predict(feats)[0]
        idx = np.argmax(proba)
        end = time.perf_counter()
        exec_time = round(end - start, 4)

        # Guardar en la base de datos
        pred_db = Prediccion(
            nombre_archivo=unique_name,
            resultado=pred,
            user_id=user_id,
            tiempo_ejecucion=exec_time
        )
        db.add(pred_db)
        db.commit()
        db.refresh(pred_db)

        return PredictionResult(
            label=pred,
            probability=round(proba[idx], 4),
            filename=unique_name,
            execution_time=exec_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

@app.get("/predicciones/usuario/{user_id}", response_model=UserPredictionsResponse)
def get_all_predictions_by_user(
        user_id: int,
        current_user_id: int = Depends(get_current_user_id),
        db: Session = Depends(get_db),
        limit: int = 100,
        offset: int = 0
):
    if current_user_id is None or current_user_id != user_id:
        raise HTTPException(status_code=403, detail="Acceso denegado")

    limit = max(1, min(limit, 1000))
    offset = max(offset, 0)

    total = db.query(Prediccion).filter(Prediccion.user_id == user_id).count()
    preds = (db.query(Prediccion)
             .filter(Prediccion.user_id == user_id)
             .order_by(Prediccion.fecha.desc())
             .offset(offset).limit(limit).all())
    items = [PrediccionResponse.from_orm(p) for p in preds]

    times = [t[0] for t in db.query(Prediccion.tiempo_ejecucion)
    .filter(Prediccion.user_id == user_id).all()]
    avg_time = round(sum(times)/len(times), 4) if times else 0.0

    epoc_count = db.query(Prediccion).filter(
        Prediccion.user_id == user_id,
        Prediccion.resultado == "COPD"
    ).count()
    saludable_count = total - epoc_count
    resumen = {
        "total_predicciones": total,
        "epoc_detectado": epoc_count,
        "saludable": saludable_count,
        "tiempo_promedio_ejecucion": avg_time,
        "porcentaje_epoc": round((epoc_count/total*100) if total else 0, 2)
    }

    return UserPredictionsResponse(total=total, predicciones=items, resumen=resumen)

@app.get("/health")
def health_check():
    return {"status": "ok", "temp_dir_exists": TEMP_DIR.exists()}
