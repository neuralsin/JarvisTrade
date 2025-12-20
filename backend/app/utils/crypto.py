"""
Spec 1: Encryption utilities for Kite API secrets using AES-256
"""
from cryptography.fernet import Fernet
from app.config import settings
import base64
import hashlib


def get_cipher():
    """
    Get Fernet cipher instance using KMS master key
    """
    # Derive a valid Fernet key from KMS_MASTER_KEY
    key = hashlib.sha256(settings.KMS_MASTER_KEY.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key)
    return Fernet(fernet_key)


def encrypt_text(plaintext: str) -> str:
    """
    Encrypt plaintext using AES-256
    """
    if not plaintext:
        return None
    cipher = get_cipher()
    return cipher.encrypt(plaintext.encode()).decode()


def decrypt_text(ciphertext: str) -> str:
    """
    Decrypt ciphertext
    """
    if not ciphertext:
        return None
    cipher = get_cipher()
    return cipher.decrypt(ciphertext.encode()).decode()
