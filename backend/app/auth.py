from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .config import settings
from .db import get_db, User

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(value: str) -> str: return pwd.hash(value)
def verify_password(value: str, hashed: str) -> bool: return pwd.verify(value, hashed)

def create_token(user: User) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    return jwt.encode({"sub": user.id, "tenant_id": user.tenant_id, "exp": exp}, settings.jwt_secret, algorithm="HS256")

def current_user(token: str = Depends(oauth), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")
    user = db.get(User, user_id)
    if not user: raise HTTPException(401, "User not found")
    return user
