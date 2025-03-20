from fastapi import FastAPI, HTTPException, UploadFile, File
import joblib
import numpy as np
import librosa
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes, puedes especificar tu frontend aquí
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
    allow_credentials=True,  # Permitir el uso de cookies y autenticación
)

# Cargar modelos preentrenados utilizando joblib
progress = ["Cargando modelos preentrenados..."]
encoder = joblib.load("encoder.pkl")
modelo = joblib.load("modelo_svc.pkl")
scaler = joblib.load("scaler.pkl")
progress.append("Modelos cargados exitosamente.")

def extraer_caracteristicas(ruta_audio, n_mfcc=13):
    progress.append("Cargando audio y extrayendo MFCCs...")
    y, sr = librosa.load(ruta_audio, sr=None)
    progress.append(f"Audio cargado con tasa de muestreo: {sr}")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    progress.append(f"MFCCs extraídos: {mfccs.shape}")
    return np.mean(mfccs, axis=1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global progress  # Permitir acceso a la lista de progreso
    progress = ["Iniciando predicción..."]

    try:
        progress.append(f"Nombre del archivo recibido: {file.filename}")

        # Guardar el archivo temporalmente
        temp_file = "temp_audio.wav"
        progress.append("Guardando archivo de audio...")
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Extraer características del audio
        progress.append("Extrayendo características del audio...")
        features = extraer_caracteristicas(temp_file)

        # Eliminar el archivo temporal
        progress.append("Eliminando archivo temporal...")
        os.remove(temp_file)

        # Preparar los datos para la predicción
        progress.append("Escalando características...")
        input_array = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_array)
        progress.append(f"Datos escalados: {scaled_data}")

        # Realizar predicción
        progress.append("Realizando predicción...")
        prediction = modelo.predict(scaled_data)[0]
        progress.append(f"Etiqueta predicha: {prediction}")

        # Calcular la probabilidad de la predicción
        progress.append("Calculando probabilidad...")
        probability = max(modelo.predict_proba(scaled_data)[0])
        progress.append(f"Probabilidad calculada: {probability}")

        # Decodificar la etiqueta predicha
        progress.append("Decodificando la etiqueta predicha...")
        predicted_label = encoder.inverse_transform([prediction])[0]
        progress.append(f"Etiqueta predicha decodificada: {predicted_label}")

        progress.append("Predicción completada con éxito.")

        return {
            "progress": progress,
            "prediction": predicted_label,
            "probability": probability
        }

    except Exception as e:
        progress.append(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail={"progress": progress, "error": str(e)})
