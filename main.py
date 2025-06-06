from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import Base, engine, SessionLocal
from models import Prediccion
from auth import get_current_user_id
from modelo_svc import modelo_svc
import librosa
import numpy as np
import time
import os
import uuid
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Configuración CORS (ajusta según origen de tu frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://id-preview--5341bf68-42ab-4af9-ba52-dd9e41fde1ae.lovable.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas en BD si no existen
Base.metadata.create_all(bind=engine)

# Modelo Pydantic para respuesta serializada
class PrediccionResponse(BaseModel):
    id: int
    nombre_archivo: str
    resultado: str
    user_id: int
    tiempo_ejecucion: float

    class Config:
        orm_mode = True

# Dependencia para obtener sesión de BD
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Función para guardar predicciones en BD
def guardar_en_db(db: Session, nombre_archivo: str, resultado: str, user_id: int, tiempo_ejecucion: float):
    pred = Prediccion(
        nombre_archivo=nombre_archivo,
        resultado=resultado,
        user_id=user_id,
        tiempo_ejecucion=tiempo_ejecucion
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)  # Actualiza el objeto con valores de BD (p.ej. id)
    return pred

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida")

    # Crear directorio temporal si no existe
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)

    # Crear nombre único para archivo
    extension = os.path.splitext(file.filename)[1] or ".wav"  # fallback a .wav si no tiene extension
    unique_filename = f"{uuid.uuid4()}{extension}"
    file_path = os.path.join(temp_dir, unique_filename)

    contenido = await file.read()
    with open(file_path, "wb") as f:
        f.write(contenido)

    # Procesar audio y extraer features
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        features = np.mean(mfccs.T, axis=0)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error procesando audio: {e}")

    start = time.time()
    try:
        resultado = modelo_svc.predict([features])[0]
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")
    end = time.time()

    # Guardar resultado en BD
    pred = guardar_en_db(db, unique_filename, resultado, user_id, end - start)

    # Borrar archivo temporal
    os.remove(file_path)

    return {
        "archivo": unique_filename,
        "resultado": resultado,
        "tiempo_ejecucion_segundos": round(end - start, 3)
    }


@app.get("/mis-predicciones", response_model=List[PrediccionResponse])
def listar_predicciones(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida")

    resultados = db.query(Prediccion).filter(Prediccion.user_id == user_id).all()
    return resultados
