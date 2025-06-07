import asyncio
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Union, Optional
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
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

class AppConfig:
    """Application configuration"""
    CORS_ORIGINS = ["https://id-preview--5341bf68-42ab-4af9-ba52-dd9e41fde1ae.lovable.app"]
    MAX_FILE_SIZE = MAX_FILE_SIZE
    SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Application shutting down")

app = FastAPI(
    title="Audio Prediction API",
    description="API para predicción de EPOC usando modelos de ML",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class PrediccionResponse(BaseModel):
    """Response model for predictions"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    nombre_archivo: str
    resultado: str
    user_id: int
    tiempo_ejecucion: float = Field(..., description="Execution time in seconds")
    fecha: str = Field(..., description="Prediction date")

    @classmethod
    def from_orm(cls, obj):
        """Custom method to handle datetime conversion"""
        return cls(
            id=obj.id,
            nombre_archivo=obj.nombre_archivo,
            resultado=obj.resultado,
            user_id=obj.user_id,
            tiempo_ejecucion=obj.tiempo_ejecucion,
            fecha=obj.fecha.isoformat() if hasattr(obj.fecha, 'isoformat') else str(obj.fecha)
        )

class PrediccionDetailResponse(BaseModel):
    """Detailed response model for individual prediction"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    nombre_archivo: str
    resultado: str
    resultado_detallado: str = Field(..., description="Detailed result description")
    user_id: int
    tiempo_ejecucion: float = Field(..., description="Execution time in seconds")
    fecha: str = Field(..., description="Prediction date")
    confianza: Optional[float] = Field(None, description="Prediction confidence if available")

class PredictionResult(BaseModel):
    """Prediction result model"""
    archivo: str = Field(..., description="Processed filename")
    resultado: str = Field(..., description="Prediction result")
    tiempo_ejecucion_segundos: float = Field(..., description="Execution time in seconds")

class UserPredictionsResponse(BaseModel):
    """Response model for user predictions list"""
    total: int = Field(..., description="Total number of predictions")
    predicciones: List[PrediccionResponse] = Field(..., description="List of predictions")
    resumen: dict = Field(..., description="Summary statistics")

# Dependencies
def get_db() -> Session:
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility Functions
def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

def convert_prediction_result(raw_result) -> str:
    """Convert model prediction to human-readable format"""
    # Convertir numpy types a Python nativo
    if isinstance(raw_result, np.integer):
        result_value = int(raw_result)
    elif isinstance(raw_result, np.floating):
        result_value = float(raw_result)
    elif isinstance(raw_result, np.ndarray):
        result_value = raw_result.item() if raw_result.size == 1 else raw_result
    else:
        result_value = raw_result

    # Convertir a resultado legible
    # Asumiendo que 1 = EPOC DETECTADO, 0 = SALUDABLE
    # Ajusta esta lógica según tu modelo
    if result_value == 1 or result_value > 0.5:
        return "EPOC DETECTADO"
    else:
        return "SALUDABLE"

def get_detailed_result_description(resultado: str) -> str:
    """Get detailed description for the result"""
    if resultado == "EPOC DETECTADO":
        return "Se ha detectado patrones característicos de EPOC en el audio analizado. Se recomienda consultar con un profesional médico para una evaluación completa."
    else:
        return "El análisis del audio no muestra patrones característicos de EPOC. Los patrones respiratorios parecen normales."

def convert_numpy_types(value) -> Union[int, float, str]:
    """Convert numpy types to native Python types"""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else str(value)
    elif isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value

async def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file to temporary directory"""
    try:
        # Generate unique filename
        file_ext = Path(file.filename).suffix or ".wav"
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = TEMP_DIR / unique_filename

        # Read and save file content
        content = await file.read()

        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )

        with open(file_path, "wb") as f:
            f.write(content)

        return file_path

    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error saving uploaded file"
        )

def extract_audio_features(file_path: Path) -> np.ndarray:
    """Extract MFCC features from audio file"""
    try:
        # Load audio with librosa
        audio, sr = librosa.load(file_path, sr=None)

        # Validate audio data
        if len(audio) == 0:
            raise ValueError("Empty audio file")

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features = np.mean(mfccs.T, axis=0)

        # Validate features
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Invalid features extracted from audio")

        return features

    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing audio file: {str(e)}"
        )

async def cleanup_file(file_path: Path) -> None:
    """Safely remove temporary file"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")

def save_prediction_to_db(
    db: Session,
    filename: str,
    result: str,
    user_id: int,
    execution_time: float
) -> Prediccion:
    """Save prediction result to database"""
    try:
        prediction = Prediccion(
            nombre_archivo=filename,
            resultado=result,
            user_id=user_id,
            tiempo_ejecucion=execution_time
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction

    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error saving prediction to database"
        )

# API Endpoints
@app.post("/predict", response_model=PredictionResult)
async def predict_audio(
    file: UploadFile = File(..., description="Audio file for prediction"),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Predict EPOC classification using trained ML model.

    - **file**: Audio file (supported formats: wav, mp3, flac, m4a, ogg)
    - **user_id**: User ID (from authentication)
    """
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Validate file
    validate_audio_file(file)

    file_path = None
    try:
        # Save uploaded file
        file_path = await save_uploaded_file(file)
        logger.info(f"Processing file: {file_path.name} for user: {user_id}")

        # Extract features
        features = extract_audio_features(file_path)

        # Make prediction
        start_time = time.perf_counter()
        try:
            raw_result = modelo_svc.predict([features])[0]
            result = convert_prediction_result(raw_result)
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error during model prediction"
            )
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Save to database
        prediction = save_prediction_to_db(
            db, file_path.name, result, user_id, execution_time
        )

        logger.info(f"Prediction completed: {result} in {execution_time:.3f}s")

        return PredictionResult(
            archivo=file_path.name,
            resultado=result,
            tiempo_ejecucion_segundos=round(execution_time, 4)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )
    finally:
        # Always cleanup temporary file
        if file_path:
            await cleanup_file(file_path)

@app.get("/predicciones/usuario/{user_id}", response_model=UserPredictionsResponse)
async def get_all_predictions_by_user(
    user_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    limit: int = 100,
    offset: int = 0
):
    """
    Get all predictions for a specific user with summary statistics.

    - **user_id**: ID of the user whose predictions to retrieve
    - **limit**: Maximum number of results (default: 100, max: 1000)
    - **offset**: Number of results to skip (default: 0)
    """
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Users can only see their own predictions (you can modify this logic as needed)
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only view your own predictions"
        )

    # Validate pagination parameters
    if limit > 1000:
        limit = 1000
    if limit < 1:
        limit = 1
    if offset < 0:
        offset = 0

    try:
        # Get total count
        total_predictions = (
            db.query(Prediccion)
            .filter(Prediccion.user_id == user_id)
            .count()
        )

        # Get predictions with pagination
        predictions_db = (
            db.query(Prediccion)
            .filter(Prediccion.user_id == user_id)
            .order_by(Prediccion.fecha.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # Convert to response model manually
        predictions = [
            PrediccionResponse.from_orm(pred) for pred in predictions_db
        ]

        # Calculate summary statistics
        epoc_count = (
            db.query(Prediccion)
            .filter(Prediccion.user_id == user_id, Prediccion.resultado == "EPOC DETECTADO")
            .count()
        )

        saludable_count = (
            db.query(Prediccion)
            .filter(Prediccion.user_id == user_id, Prediccion.resultado == "SALUDABLE")
            .count()
        )

        # Calculate average execution time
        avg_execution_time = (
            db.query(Prediccion.tiempo_ejecucion)
            .filter(Prediccion.user_id == user_id)
            .all()
        )

        avg_time = sum(p.tiempo_ejecucion for p in avg_execution_time) / len(avg_execution_time) if avg_execution_time else 0

        resumen = {
            "total_predicciones": total_predictions,
            "epoc_detectado": epoc_count,
            "saludable": saludable_count,
            "tiempo_promedio_ejecucion": round(avg_time, 4),
            "porcentaje_epoc": round((epoc_count / total_predictions * 100) if total_predictions > 0 else 0, 2)
        }

        return UserPredictionsResponse(
            total=total_predictions,
            predicciones=predictions,
            resumen=resumen
        )

    except Exception as e:
        logger.error(f"Database error retrieving predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving predictions"
        )

@app.get("/predicciones/{prediccion_id}", response_model=PrediccionDetailResponse)
async def get_prediction_detail(
    prediccion_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific prediction.

    - **prediccion_id**: ID of the prediction to retrieve
    """
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        # Get the prediction
        prediction = (
            db.query(Prediccion)
            .filter(Prediccion.id == prediccion_id)
            .first()
        )

        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )

        # Check if user has access to this prediction
        if prediction.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view your own predictions"
            )

        # Create detailed response
        detailed_response = PrediccionDetailResponse(
            id=prediction.id,
            nombre_archivo=prediction.nombre_archivo,
            resultado=prediction.resultado,
            resultado_detallado=get_detailed_result_description(prediction.resultado),
            user_id=prediction.user_id,
            tiempo_ejecucion=prediction.tiempo_ejecucion,
            fecha=prediction.fecha.isoformat() if hasattr(prediction.fecha, 'isoformat') else str(prediction.fecha)
        )

        return detailed_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error retrieving prediction detail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving prediction details"
        )

@app.get("/mis-predicciones", response_model=List[PrediccionResponse])
async def get_user_predictions(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    limit: int = 100,
    offset: int = 0
):
    """
    Get user's prediction history with pagination.

    - **limit**: Maximum number of results (default: 100, max: 1000)
    - **offset**: Number of results to skip (default: 0)
    """
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Validate pagination parameters
    if limit > 1000:
        limit = 1000
    if limit < 1:
        limit = 1
    if offset < 0:
        offset = 0

    try:
        predictions_db = (
            db.query(Prediccion)
            .filter(Prediccion.user_id == user_id)
            .order_by(Prediccion.fecha.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # Convert to response model manually
        predictions = [
            PrediccionResponse.from_orm(pred) for pred in predictions_db
        ]
        
        return predictions
        
    except Exception as e:
        logger.error(f"Database error retrieving predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving predictions"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "temp_dir_exists": TEMP_DIR.exists()
    }