import asyncio
import logging
import os
import uuid
import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
from fastapi import (
    FastAPI, UploadFile, File, Depends,
    HTTPException, status, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import Base, engine, SessionLocal
from models import Prediccion
from auth import get_current_user_id
from modelo_svc import modelo_svc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
TEMP_DIR = Path("temp_audio")
N_MFCC = 13
executor = ThreadPoolExecutor()

class AppConfig:
    CORS_ORIGINS = ["https://id-preview--5341bf68-42ab-4af9-ba52-dd9e41fde1ae.lovable.app"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    TEMP_DIR.mkdir(exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Application started successfully")
    yield
    logger.info("Application shutting down")

app = FastAPI(
    title="Audio Prediction API",
    version="1.0.1",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class PredictionResult(BaseModel):
    archivo: str = Field(..., description="Processed filename")
    resultado: str = Field(..., description="Prediction result")
    tiempo_ejecucion_segundos: float = Field(..., description="Execution time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "archivo": "123e4567-a89b-12d3-a456-426614174000.wav",
                "resultado": "SALUDABLE",
                "tiempo_ejecucion_segundos": 0.2345
            }
        }

class PrediccionResponse(BaseModel):
    id: int
    nombre_archivo: str
    resultado: str
    user_id: int
    tiempo_ejecucion: float = Field(..., description="Execution time in seconds")
    fecha: str = Field(..., description="Prediction date in ISO format")

    class Config:
        orm_mode = True

class UserPredictionsResponse(BaseModel):
    total: int = Field(..., description="Total number of predictions")
    predicciones: List[PrediccionResponse]
    resumen: dict

# Dependency
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utilities
def validate_audio_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Filename is required")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {ext}. Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

async def save_uploaded_file(file: UploadFile) -> Path:
    file_ext = Path(file.filename).suffix or ".wav"
    unique_name = f"{uuid.uuid4()}{file_ext}"
    out_path = TEMP_DIR / unique_name
    size = 0
    try:
        with open(out_path, 'wb') as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Max {MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                buffer.write(chunk)
        return out_path
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error saving uploaded file")

def extract_audio_features(file_path: Path) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=None)
    if audio.size == 0:
        raise ValueError("Empty audio file")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    feats = np.mean(mfccs.T, axis=0)
    if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
        raise ValueError("Invalid features extracted")
    return feats

async def cleanup_file(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.warning(f"Cleanup failed for {path}: {e}")

def save_prediction_to_db(db: Session, filename: str, result: str, user_id: int, exec_time: float) -> Prediccion:
    pred = Prediccion(
        nombre_archivo=filename,
        resultado=result,
        user_id=user_id,
        tiempo_ejecucion=exec_time
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred

# Endpoints
@app.post("/predict", response_model=PredictionResult)
async def predict_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    validate_audio_file(file)
    file_path = await save_uploaded_file(file)
    background_tasks.add_task(cleanup_file, file_path)

    loop = asyncio.get_event_loop()
    try:
        features = await loop.run_in_executor(executor, extract_audio_features, file_path)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    start = time.perf_counter()
    try:
        probs = modelo_svc.predict_proba([features])[0]
        if probs[1] > 0.5:
            result = "EPOC DETECTADO"
            confianza = float(probs[1])
        else:
            result = "SALUDABLE"
            confianza = float(probs[0])
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error during model prediction")
    end = time.perf_counter()

    exec_time = end - start
    save_prediction_to_db(db, file_path.name, result, user_id, exec_time)
    logger.info(f"Predicted {result} ({confianza:.2f}) in {exec_time:.3f}s for user {user_id}")

    return PredictionResult(
        archivo=file_path.name,
        resultado=result,
        tiempo_ejecucion_segundos=round(exec_time, 4)
    )

@app.get("/predicciones/usuario/{user_id}", response_model=UserPredictionsResponse)
async def get_all_predictions_by_user(
    user_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    limit: int = 100,
    offset: int = 0
):
    if current_user_id is None or current_user_id != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Access denied")

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
        Prediccion.resultado == "EPOC DETECTADO"
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
async def health_check():
    return {"status": "healthy", "timestamp": time.time(), "temp_dir_exists": TEMP_DIR.exists()}
