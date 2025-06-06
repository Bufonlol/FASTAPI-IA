from fastapi import FastAPI, UploadFile, File, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine, SessionLocal
from models import Prediccion
from auth import get_current_user_id  # Aquí debes tener la versión actualizada
from modelo_svc import modelo_svc
import librosa
import numpy as np
import time
from auth import SECRET_KEY, ALGORITHM
from auth import get_current_user_id, oauth2_scheme
from jose import JWTError, jwt


app = FastAPI()

# CORS (ajusta según origen de tu frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://id-preview--5341bf68-42ab-4af9-ba52-dd9e41fde1ae.lovable.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas
Base.metadata.create_all(bind=engine)


# Función para guardar predicciones
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

# Dependencia adaptada (la puedes poner en auth.py)
async def get_current_user_id(request: Request, token: str = Depends(oauth2_scheme)):
    if request.method == "OPTIONS":
        # Preflight no requiere token
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("userId")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token inválido: userId faltante")
        return user_id
    except JWTError as e:
        print("Error al decodificar token:", e)
        raise HTTPException(status_code=401, detail="Token inválido")

@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: int = Depends(get_current_user_id)):
    if user_id is None:
        # En caso de preflight, no procesamos
        raise HTTPException(status_code=401, detail="Autenticación requerida")

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
    if user_id is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida")

    db = SessionLocal()
    resultados = db.query(Prediccion).filter(Prediccion.user_id == user_id).all()
    db.close()
    return resultados
