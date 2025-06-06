from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "7Xp4Mv2bT5qYt8wZzR9kA1cF3eD6gHjJ7LmN0oP4iK5lB8uV2nC9xQ3wE5rT6yU"
ALGORITHM = "HS384"

def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS384"])
        user_id = payload.get("userId")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token inválido: userId faltante")
        return user_id
    except JWTError as e:
        print("Error al decodificar token:", e)
        raise HTTPException(status_code=401, detail="Token inválido")
