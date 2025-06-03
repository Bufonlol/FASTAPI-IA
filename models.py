from sqlalchemy import Column, Integer, String, DateTime, Float
from datetime import datetime
from database import Base

class Prediccion(Base):
    __tablename__ = "predicciones"

    id = Column(Integer, primary_key=True, index=True)
    nombre_archivo = Column(String, index=True)
    resultado = Column(String)
    user_id = Column(Integer, index=True)
    fecha = Column(DateTime, default=datetime.utcnow)
    tiempo_ejecucion = Column(Float)
