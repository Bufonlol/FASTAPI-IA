# Usar una imagen base de Python
FROM python:3.10

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY main.py /app/
COPY encoder.pkl /app/
COPY modelo_svc.pkl /app/
COPY scaler.pkl /app/
COPY requirements.txt /app/

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que FastAPI correrá
EXPOSE 8000

# Comando para iniciar la API con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
