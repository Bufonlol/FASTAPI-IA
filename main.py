from fastapi import FastAPI, HTTPException, UploadFile, File
import joblib
import numpy as np
import librosa
import os

app = FastAPI()

# Cargar modelos preentrenados utilizando joblib
encoder = joblib.load("encoder.pkl")
modelo = joblib.load("modelo_svc.pkl")
scaler = joblib.load("scaler.pkl")

def extraer_caracteristicas(ruta_audio, n_mfcc=13):
    # Cargar el audio y extraer los MFCCs
    y, sr = librosa.load(ruta_audio, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Guardar el archivo temporalmente
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Extraer características del audio
        features = extraer_caracteristicas(temp_file)
        os.remove(temp_file)  # Limpiar el archivo temporal

        # Preparar los datos para la predicción
        input_array = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_array)
        prediction = modelo.predict(scaled_data)[0]
        probability = max(modelo.predict_proba(scaled_data)[0])
        predicted_label = encoder.inverse_transform([prediction])[0]

        return {"prediction": predicted_label, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
