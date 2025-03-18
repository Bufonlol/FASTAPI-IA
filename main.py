from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Cargar modelos preentrenados
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("modelo_svc.pkl", "rb") as f:
    modelo = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Definir el esquema de entrada
class PredictionInput(BaseModel):
    audio_features: list[float]

# Endpoint para realizar predicciones
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convertir la lista en un array y ajustarlo para el modelo
        input_array = np.array(input_data.audio_features).reshape(1, -1)
        scaled_data = scaler.transform(input_array)
        prediction = modelo.predict(scaled_data)[0]
        probability = max(modelo.predict_proba(scaled_data)[0])
        # Convertir la predicción al formato original
        predicted_label = encoder.inverse_transform([prediction])[0]
        return {"prediction": predicted_label, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
