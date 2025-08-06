from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://sirade_bd_fhxm_user:kY6No7QeKHL0t1BelcAa6VpjSUskT7rn@dpg-d29dd1mr433s7398f9ng-a/sirade_bd_fhxm"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
