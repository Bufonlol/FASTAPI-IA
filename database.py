from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://sirade_bd_xq32_user:DbjbU7lc3tI0ZeUBBWvXcldh32aNHrlX@dpg-d4ges6ndiees73fo3lqg-a.oregon-postgres.render.com/sirade_bd_xq32"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
