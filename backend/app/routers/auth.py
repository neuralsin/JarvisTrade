"""
Authentication routes - login, register, Kite OAuth
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.db.database import get_db
from app.db.models import User
from app.config import settings
from app.utils.crypto import encrypt_text, decrypt_text
import logging
import hashlib
import base64

logger = logging.getLogger(__name__)
router = APIRouter()


def _prepare_password(password: str) -> str:
    """
    Pre-hash password with SHA256 and encode as base64 before bcrypt.
    This avoids bcrypt's 72-byte limitation while maintaining security.
    The base64 encoded SHA256 hash is 44 characters, well within the 72-byte limit.
    """
    password_hash = hashlib.sha256(password.encode('utf-8')).digest()
    return base64.b64encode(password_hash).decode('ascii')

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# JWT settings
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


class UserRegister(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class KiteCredentials(BaseModel):
    api_key: str
    api_secret: str


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str):
    """
    Decode and validate JWT token.
    Used by WebSocket endpoint for authentication.
    
    Returns:
        dict: Decoded token payload
    
    Raises:
        JWTError: If token is invalid or expired
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user


@router.post("/register", response_model=Token)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register new user
    """
    # Check if user exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = pwd_context.hash(_prepare_password(user_data.password))
    new_user = User(
        email=user_data.email,
        password_hash=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"New user registered: {user_data.email}")
    
    # Create access token
    access_token = create_access_token(data={"sub": str(new_user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Login user
    """
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not pwd_context.verify(_prepare_password(form_data.password), user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token(data={"sub": str(user.id)})
    logger.info(f"User logged in: {user.email}")
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/kite/credentials")
async def save_kite_credentials(
    credentials: KiteCredentials,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Save encrypted Kite API credentials
    """
    current_user.kite_api_key_encrypted = encrypt_text(credentials.api_key)
    current_user.kite_api_secret_encrypted = encrypt_text(credentials.api_secret)
    db.commit()
    
    logger.info(f"Kite credentials saved for user: {current_user.email}")
    return {"status": "success", "message": "Kite credentials saved"}


@router.get("/kite/login-url")
async def get_kite_login_url(current_user: User = Depends(get_current_user)):
    """
    Spec 18: Kite OAuth flow - generate login URL
    """
    if not current_user.kite_api_key_encrypted:
        raise HTTPException(status_code=400, detail="Kite API key not configured")
    
    api_key = decrypt_text(current_user.kite_api_key_encrypted)
    login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&redirect_params={current_user.id}"
    
    return {"login_url": login_url}


@router.get("/kite/callback")
async def kite_callback(
    request_token: str,
    status: str,
    db: Session = Depends(get_db)
):
    """
    Complete Kite OAuth flow - exchange request_token for access_token
    
    Flow:
    1. User gets redirected here after Kite login
    2. We receive request_token from Kite
    3. Exchange request_token for access_token using API secret
    4. Store encrypted access_token in database
    """
    try:
        if status != "success":
            logger.error(f"Kite OAuth failed with status: {status}")
            raise HTTPException(status_code=400, detail="Kite authentication failed")
        
        logger.info(f"Kite OAuth callback received: request_token={request_token[:10]}..., status={status}")
        
        # Get the first user (for now - in production you'd extract user_id from state param)
        user = db.query(User).first()
        if not user:
            logger.error("No user found in database")
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if user has Kite credentials stored
        if not user.kite_api_key_encrypted or not user.kite_api_secret_encrypted:
            logger.error(f"User {user.email} has no Kite credentials")
            raise HTTPException(
                status_code=400,
                detail="Kite API credentials not found. Please save them first."
            )
        
        # Decrypt user's API credentials
        api_key = decrypt_text(user.kite_api_key_encrypted)
        api_secret = decrypt_text(user.kite_api_secret_encrypted)
        
        # Initialize Kite and exchange request_token for access_token
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        
        logger.info(f"Generating Kite session for user {user.email}")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        # Store encrypted access token
        user.kite_access_token_encrypted = encrypt_text(access_token)
        db.commit()
        
        logger.info(f"✅ Kite access token stored successfully for {user.email}")
        
        # Return success page or redirect to frontend
        return {
            "status": "success",
            "message": "Kite authentication successful! You can now close this window.",
            "user_email": user.email
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Kite OAuth callback error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete Kite authentication: {str(e)}"
        )



@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information
    """
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "has_kite_credentials": bool(current_user.kite_api_key_encrypted),
        "auto_execute": current_user.auto_execute
    }
