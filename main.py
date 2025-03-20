@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global progress
    progress = ["Iniciando predicción..."]

    try:
        progress.append(f"Nombre del archivo recibido: {file.filename}")
        print(f"Nombre del archivo recibido: {file.filename}")  # Log para depuración

        # Guardar el archivo temporalmente
        temp_file = "temp_audio.wav"
        progress.append("Guardando archivo de audio...")
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        print(f"Archivo guardado como: {temp_file}")  # Log para depuración

        # Extraer características del audio
        progress.append("Extrayendo características del audio...")
        features = extraer_caracteristicas(temp_file)
        print(f"Características extraídas: {features}")  # Log para depuración

        # Eliminar el archivo temporal
        progress.append("Eliminando archivo temporal...")
        os.remove(temp_file)
        print("Archivo temporal eliminado.")  # Log para depuración

        # Preparar los datos para la predicción
        progress.append("Escalando características...")
        input_array = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_array)
        progress.append(f"Datos escalados: {scaled_data}")
        print(f"Datos escalados: {scaled_data}")  # Log para depuración

        # Realizar predicción
        progress.append("Realizando predicción...")
        prediction = modelo.predict(scaled_data)[0]
        progress.append(f"Etiqueta predicha: {prediction}")
        print(f"Etiqueta predicha: {prediction}")  # Log para depuración

        # Calcular la probabilidad de la predicción
        progress.append("Calculando probabilidad...")
        probability = max(modelo.predict_proba(scaled_data)[0])
        progress.append(f"Probabilidad calculada: {probability}")
        print(f"Probabilidad calculada: {probability}")  # Log para depuración

        # Decodificar la etiqueta predicha
        progress.append("Decodificando la etiqueta predicha...")
        predicted_label = encoder.inverse_transform([prediction])[0]
        progress.append(f"Etiqueta predicha decodificada: {predicted_label}")
        print(f"Etiqueta predicha decodificada: {predicted_label}")  # Log para depuración

        progress.append("Predicción completada con éxito.")
        return {
            "progress": progress,
            "prediction": predicted_label,
            "probability": probability
        }

    except Exception as e:
        progress.append(f"Error: {str(e)}")
        print(f"Error: {str(e)}")  # Log para depuración
        raise HTTPException(status_code=500, detail={"progress": progress, "error": str(e)})
