from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "7Xp4Mv2bT5qYt8wZzR9kA1cF3eD6gHjJ7LmN0oP4iK5lB8uV2nC9xQ3wE5rT6yU"
ALGORITHM = "HS256"

def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("userId")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token inválido: userId faltante")
        return int(user_id)
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")
