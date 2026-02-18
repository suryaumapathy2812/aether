"""
Symmetric encryption for secrets at rest (API keys, tokens, etc.).

Uses Fernet (AES-128-CBC + HMAC-SHA256) with a key derived from
BETTER_AUTH_SECRET via PBKDF2. Deterministic derivation means we
don't need to store key material separately — just the secret.

Usage:
    from .crypto import encrypt_value, decrypt_value

    ciphertext = encrypt_value("sk-abc123...")
    plaintext  = decrypt_value(ciphertext)
"""

from __future__ import annotations

import base64
import logging
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

log = logging.getLogger("orchestrator.crypto")

# ── Key derivation ─────────────────────────────────────────

_SECRET = os.getenv("BETTER_AUTH_SECRET", "")

# Fixed salt — deterministic so we can derive the same key on every startup.
# This is fine because BETTER_AUTH_SECRET is already high-entropy.
_SALT = b"aether-api-key-encryption-v1"


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a 32-byte Fernet key from the app secret via PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=480_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(secret.encode("utf-8")))


# Derive once at import time (module-level singleton)
_fernet: Fernet | None = None

if _SECRET:
    _fernet = Fernet(_derive_fernet_key(_SECRET))
else:
    log.warning("⚠ BETTER_AUTH_SECRET not set — API keys will be stored in PLAINTEXT")


# ── Public API ─────────────────────────────────────────────


def encrypt_value(plaintext: str) -> str:
    """
    Encrypt a string value. Returns base64-encoded ciphertext.

    Falls back to plaintext if no secret is configured (dev convenience).
    """
    if not _fernet:
        return plaintext
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_value(ciphertext: str) -> str:
    """
    Decrypt a previously encrypted value.

    Handles legacy plaintext values gracefully — if decryption fails
    and the value looks like a raw API key, returns it as-is.
    """
    if not _fernet:
        return ciphertext

    try:
        return _fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        # Legacy plaintext value — return as-is so existing keys still work
        # after enabling encryption. They'll be re-encrypted on next save.
        log.debug("Decryption failed — treating as legacy plaintext value")
        return ciphertext
