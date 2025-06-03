from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://sirade_bd_user:HSRipcmtYKo5QsP7sr5D9IDaYz6zOYIk@dpg-d0vh9ie3jp1c73e6usb0-a.oregon-postgres.render.com/sirade_bd"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
