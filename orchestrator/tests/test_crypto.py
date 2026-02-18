"""Tests for the crypto module — Fernet encryption for API keys at rest."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestDeriveKey:
    """Key derivation from BETTER_AUTH_SECRET."""

    def test_derive_produces_valid_fernet_key(self):
        """Derived key should be 44 bytes (base64-encoded 32 bytes)."""
        from src.crypto import _derive_fernet_key

        key = _derive_fernet_key("test-secret-123")
        assert len(key) == 44  # base64url of 32 bytes

    def test_derive_deterministic(self):
        """Same secret always produces the same key."""
        from src.crypto import _derive_fernet_key

        k1 = _derive_fernet_key("my-secret")
        k2 = _derive_fernet_key("my-secret")
        assert k1 == k2

    def test_derive_different_secrets_different_keys(self):
        """Different secrets produce different keys."""
        from src.crypto import _derive_fernet_key

        k1 = _derive_fernet_key("secret-a")
        k2 = _derive_fernet_key("secret-b")
        assert k1 != k2


class TestEncryptDecrypt:
    """Encrypt/decrypt roundtrip with a real Fernet instance."""

    def _make_module_with_secret(self, secret: str):
        """Import crypto module with a specific BETTER_AUTH_SECRET."""
        import importlib
        import os

        with patch.dict(os.environ, {"BETTER_AUTH_SECRET": secret}):
            import src.crypto as mod

            importlib.reload(mod)
            return mod

    def _make_module_without_secret(self):
        """Import crypto module with no BETTER_AUTH_SECRET."""
        import importlib
        import os

        env = os.environ.copy()
        env.pop("BETTER_AUTH_SECRET", None)
        with patch.dict(os.environ, env, clear=True):
            import src.crypto as mod

            importlib.reload(mod)
            return mod

    def test_roundtrip(self):
        """encrypt then decrypt returns original plaintext."""
        mod = self._make_module_with_secret("roundtrip-test-secret")
        try:
            plaintext = "sk-abc123-very-secret-key"
            ciphertext = mod.encrypt_value(plaintext)
            assert ciphertext != plaintext
            assert ciphertext.startswith("gAAAAA")  # Fernet prefix
            assert mod.decrypt_value(ciphertext) == plaintext
        finally:
            # Restore module state
            self._make_module_without_secret()

    def test_ciphertext_is_different_each_time(self):
        """Fernet uses a random IV, so encrypting the same value twice gives different ciphertext."""
        mod = self._make_module_with_secret("iv-test-secret")
        try:
            plaintext = "sk-same-key"
            c1 = mod.encrypt_value(plaintext)
            c2 = mod.encrypt_value(plaintext)
            assert c1 != c2
            # But both decrypt to the same value
            assert mod.decrypt_value(c1) == plaintext
            assert mod.decrypt_value(c2) == plaintext
        finally:
            self._make_module_without_secret()

    def test_no_secret_passthrough(self):
        """Without BETTER_AUTH_SECRET, encrypt/decrypt are identity functions."""
        mod = self._make_module_without_secret()
        assert mod.encrypt_value("sk-plain") == "sk-plain"
        assert mod.decrypt_value("sk-plain") == "sk-plain"

    def test_legacy_plaintext_fallback(self):
        """Decrypting a non-Fernet value returns it as-is (legacy support)."""
        mod = self._make_module_with_secret("legacy-test-secret")
        try:
            # This is not valid Fernet ciphertext — should fall back gracefully
            legacy = "sk-old-plaintext-key"
            assert mod.decrypt_value(legacy) == legacy
        finally:
            self._make_module_without_secret()

    def test_wrong_secret_returns_plaintext_fallback(self):
        """Decrypting with a different secret falls back to returning ciphertext as-is."""
        mod1 = self._make_module_with_secret("secret-one")
        ciphertext = mod1.encrypt_value("sk-secret-data")

        mod2 = self._make_module_with_secret("secret-two")
        try:
            # Wrong key can't decrypt — should return ciphertext as-is (legacy fallback)
            result = mod2.decrypt_value(ciphertext)
            assert result == ciphertext
        finally:
            self._make_module_without_secret()
