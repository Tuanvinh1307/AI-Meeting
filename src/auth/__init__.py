"""
Package Xac thuc & Bao mat (JWT + Bcrypt).
"""

from src.auth.security import (
    authenticate_user_and_upgrade_hash,
    create_access_token,
    decode_access_token,
    get_current_user,
    get_current_user_audio,
    hash_password,
    is_bcrypt_hash,
    require_admin,
    require_admin_audio,
    verify_password,
)

__all__ = [
    "authenticate_user_and_upgrade_hash",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "get_current_user_audio",
    "hash_password",
    "is_bcrypt_hash",
    "require_admin",
    "require_admin_audio",
    "verify_password",
]
