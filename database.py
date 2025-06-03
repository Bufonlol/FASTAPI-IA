from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://sirade_database_0s0b_user:yPAIgoHn2R2kt3oIJkhkJLDLKPkOOqPd@dpg-cv8kep0gph6c73acia50-a.oregon-postgres.render.com:5432/sirade_database_0s0b"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
