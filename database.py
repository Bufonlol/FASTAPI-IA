from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://sirade_bd_q5nv_user:1yseUfaHhS0dvGMjvVBe8JJnyeRUnkm4@dpg-d1jskvp5pdvs73fcetsg-a.oregon-postgres.render.com/sirade_bd_q5nv"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
