from app.db.database import SessionLocal
from app.db.models import User
from passlib.context import CryptContext
import hashlib
import base64

pwd_context = CryptContext(schemes=['bcrypt'])

def prepare_password(password):
    password_hash = hashlib.sha256(password.encode('utf-8')).digest()
    return base64.b64encode(password_hash).decode('ascii')

db = SessionLocal()
user = db.query(User).filter(User.email == 'test@example.com').first()

if user:
    # Reset password
    user.password_hash = pwd_context.hash(prepare_password('testpass123'))
    user.auto_execute = True
    user.paper_trading_enabled = True
    db.commit()
    print(f'✓ Password reset for {user.email}')
    print(f'  auto_execute: {user.auto_execute}')
    print(f'  paper_trading_enabled: {user.paper_trading_enabled}')
else:
    print('✗ User not found')
    
db.close()
