import asyncio
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Union
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
    description="API para predicción de audio usando modelos de ML",
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

class PredictionResult(BaseModel):
    """Prediction result model"""
    archivo: str = Field(..., description="Processed filename")
    resultado: Union[int, str] = Field(..., description="Prediction result")
    tiempo_ejecucion_segundos: float = Field(..., description="Execution time in seconds")

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
    result: Union[int, str], 
    user_id: int, 
    execution_time: float
) -> Prediccion:
    """Save prediction result to database"""
    try:
        prediction = Prediccion(
            nombre_archivo=filename,
            resultado=str(result),
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
    Predict audio classification using trained ML model.
    
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
            result = convert_numpy_types(raw_result)
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
        predictions = (
            db.query(Prediccion)
            .filter(Prediccion.user_id == user_id)
            .order_by(Prediccion.fecha.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        
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