from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine, SessionLocal
from models import Prediccion
from auth import get_current_user_id
from modelo_svc import modelo_svc
import librosa
import numpy as np
import time

app = FastAPI()

# CORS (ajusta según origen de tu frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas
Base.metadata.create_all(bind=engine)

# Guardar predicción
def guardar_en_db(nombre_archivo, resultado, user_id, tiempo_ejecucion):
    db = SessionLocal()
    pred = Prediccion(
        nombre_archivo=nombre_archivo,
        resultado=resultado,
        user_id=user_id,
        tiempo_ejecucion=tiempo_ejecucion
    )
    db.add(pred)
    db.commit()
    db.close()

@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: int = Depends(get_current_user_id)):
    contenido = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contenido)

    audio, sr = librosa.load(file.filename, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    features = np.mean(mfccs.T, axis=0)

    start = time.time()
    resultado = modelo_svc.predict([features])[0]
    end = time.time()

    guardar_en_db(file.filename, resultado, user_id, end - start)

    return {
        "archivo": file.filename,
        "resultado": resultado,
        "tiempo_ejecucion_segundos": round(end - start, 3)
    }

@app.get("/mis-predicciones")
def listar_predicciones(user_id: int = Depends(get_current_user_id)):
    db = SessionLocal()
    resultados = db.query(Prediccion).filter(Prediccion.user_id == user_id).all()
    db.close()
    return resultados
