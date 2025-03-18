from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar modelos preentrenados utilizando joblib
try:
    encoder = joblib.load("encoder.pkl")
    modelo = joblib.load("modelo_svc.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Error al cargar los modelos: {e}")

# Definir el esquema de entrada para la predicción
class PredictionInput(BaseModel):
    audio_features: list[float]

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convertir la lista de características en un array y reformatearlo
        input_array = np.array(input_data.audio_features).reshape(1, -1)
        scaled_data = scaler.transform(input_array)

        # Realizar la predicción y obtener la probabilidad asociada
        prediction = modelo.predict(scaled_data)[0]
        probability = max(modelo.predict_proba(scaled_data)[0])

        # Convertir la predicción a la etiqueta original
        predicted_label = encoder.inverse_transform([prediction])[0]

        return {"prediction": predicted_label, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
