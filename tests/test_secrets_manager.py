"""
Tests für Secrets Management Module

Run: pytest tests/test_secrets_manager.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.config.secrets_manager import (
    BaseSecretsManager,
    EnvironmentSecretsManager,
    VaultSecretsManager,
    AWSSecretsManager,
    Secrets,
    SecretMetadata,
    create_secrets_manager,
    get_secrets,
    get_binance_credentials,
    get_telegram_credentials,
    reset_secrets_manager,
)

from src.config import (
    get_secret_key,
    get_admin_credentials,
    get_kucoin_credentials,
    get_bybit_credentials,
    get_github_credentials,
    get_ably_key,
    get_colab_token,
)


class TestSecretsDataclass:
    """Tests für Secrets Dataclass."""

    def test_secrets_default_values(self):
        """Test Default-Werte."""
        secrets = Secrets()
        assert secrets.binance_api_key == ""
        assert secrets.binance_api_secret == ""
        assert secrets.telegram_bot_token == ""
        assert secrets.telegram_chat_id == ""
        assert secrets.github_token is None

    def test_is_binance_configured(self):
        """Test Binance Konfigurations-Check."""
        secrets = Secrets(binance_api_key="test_key", binance_api_secret="test_secret")
        assert secrets.is_binance_configured() is True

        secrets_empty = Secrets()
        assert secrets_empty.is_binance_configured() is False

        secrets_partial = Secrets(binance_api_key="test_key")
        assert secrets_partial.is_binance_configured() is False

    def test_is_telegram_configured(self):
        """Test Telegram Konfigurations-Check."""
        secrets = Secrets(telegram_bot_token="token", telegram_chat_id="chat_id")
        assert secrets.is_telegram_configured() is True

    def test_mask_sensitive(self):
        """Test Secret Masking."""
        secrets = Secrets(
            binance_api_key="test_key_12345",
            binance_api_secret="secret_abc",
            telegram_bot_token="bot_token",
            telegram_chat_id="123456",
            github_token="ghp_abc",
        )

        masked = secrets.mask_sensitive()

        # Prüfe dass die letzten 4 Zeichen sichtbar sind
        assert masked["binance_api_key"].endswith("2345")
        assert masked["binance_api_secret"].endswith("bc")
        assert masked["telegram_bot_token"].endswith("oken")
        assert masked["telegram_chat_id"] == "123456"  # Chat IDs werden nicht maskiert
        assert masked["github_token"].endswith("bc")

        # Prüfe dass keine echten Secrets sichtbar sind
        assert "test_key" not in masked["binance_api_key"]
        assert "secret" not in masked["binance_api_secret"]
        assert "bot_token" not in masked["telegram_bot_token"]


class TestEnvironmentSecretsManager:
    """Tests für EnvironmentSecretsManager."""

    def test_is_available(self):
        """Test dass Env Manager immer verfügbar ist."""
        manager = EnvironmentSecretsManager()
        assert manager.is_available() is True

    def test_get_from_env_var(self):
        """Test dass Secrets aus ENV Vars gelesen werden."""
        with patch.dict(
            os.environ,
            {"BINANCE_API_KEY": "env_api_key", "BINANCE_API_SECRET": "env_api_secret"},
        ):
            manager = EnvironmentSecretsManager()
            api_key = manager.get("binance/api_key")
            api_secret = manager.get("binance/api_secret")

            assert api_key == "env_api_key"
            assert api_secret == "env_api_secret"

    def test_get_with_default(self):
        """Test Default-Wert wenn ENV Var nicht gesetzt."""
        manager = EnvironmentSecretsManager()
        result = manager.get("nonexistent/key", default="default_value")
        assert result == "default_value"

    def test_get_all(self):
        """Test get_all() Methode."""
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "key",
                "BINANCE_API_SECRET": "secret",
                "TELEGRAM_BOT_TOKEN": "bot_token",
                "TELEGRAM_CHAT_ID": "chat123",
            },
        ):
            manager = EnvironmentSecretsManager()
            secrets = manager.get_all()

            assert secrets.binance_api_key == "key"
            assert secrets.binance_api_secret == "secret"
            assert secrets.telegram_bot_token == "bot_token"
            assert secrets.telegram_chat_id == "chat123"

    def test_get_binance_credentials(self):
        """Test get_binance_credentials() Tuple."""
        with patch.dict(
            os.environ,
            {"BINANCE_API_KEY": "test_key", "BINANCE_API_SECRET": "test_secret"},
        ):
            manager = EnvironmentSecretsManager()
            api_key, api_secret = manager.get_binance_credentials()

            assert api_key == "test_key"
            assert api_secret == "test_secret"

    def test_cache_functionality(self):
        """Test dass Caching funktioniert."""
        with patch.dict(os.environ, {"BINANCE_API_KEY": "cached_key"}):
            manager = EnvironmentSecretsManager()

            # Erster Aufruf
            key1 = manager.get("binance/api_key")
            # Zweiter Aufruf sollte aus Cache kommen
            key2 = manager.get("binance/api_key")

            assert key1 == key2 == "cached_key"

    def test_reload_clears_cache(self):
        """Test dass reload() Cache leert."""
        # Ensure env is clean first
        original = os.environ.get("BINANCE_API_KEY")
        try:
            os.environ["BINANCE_API_KEY"] = "initial_key"

            manager = EnvironmentSecretsManager()

            # First call - populates cache
            value1 = manager.get("binance/api_key")
            assert value1 == "initial_key"

            # Second call - should be cached
            value2 = manager.get("binance/api_key")
            assert value2 == "initial_key"

            # Change the environment
            os.environ["BINANCE_API_KEY"] = "new_key"

            # Before reload - should still have old value (from cache)
            value3 = manager.get("binance/api_key")
            assert value3 == "initial_key", f"Expected cached value, got {value3}"

            # After reload - should have new value
            manager.reload()
            value4 = manager.get("binance/api_key")
            assert value4 == "new_key", f"Expected new value after reload, got {value4}"
        finally:
            # Cleanup
            if original is not None:
                os.environ["BINANCE_API_KEY"] = original
            elif "BINANCE_API_KEY" in os.environ:
                del os.environ["BINANCE_API_KEY"]


class TestVaultSecretsManager:
    """Tests für VaultSecretsManager."""

    def test_vault_not_available_without_hvac(self):
        """Test Vault nicht verfügbar wenn hvac nicht installiert."""
        with patch("src.config.secrets_manager.hvac", None):
            manager = VaultSecretsManager()
            assert manager.is_available() is False

    def test_vault_fallback_to_env(self):
        """Test dass Vault auf ENV fall-back wenn nicht verfügbar."""
        with patch.dict(os.environ, {"BINANCE_API_KEY": "fallback_key"}):
            # Mock hvac um nicht verfügbar zu simulieren
            with patch("src.config.secrets_manager.hvac", None):
                manager = VaultSecretsManager()
                # Sollte auf ENV zurückfallen
                assert manager.get("binance/api_key") == "fallback_key"


class TestAWSSecretsManager:
    """Tests für AWSSecretsManager."""

    def test_aws_not_available_without_boto3(self):
        """Test AWS nicht verfügbar wenn boto3 nicht installiert."""
        with patch("src.config.secrets_manager.boto3", None):
            manager = AWSSecretsManager()
            assert manager.is_available() is False


class TestFactoryFunctions:
    """Tests für Factory Funktionen."""

    def test_create_secrets_manager_env(self):
        """Test create_secrets_manager mit env."""
        manager = create_secrets_manager("env")
        assert isinstance(manager, EnvironmentSecretsManager)

    def test_get_secrets_uses_env_by_default(self):
        """Test dass get_secrets() Env Manager verwendet."""
        reset_secrets_manager()

        with patch.dict(os.environ, {"BINANCE_API_KEY": "factory_test_key"}):
            secrets = get_secrets()
            assert secrets.binance_api_key == "factory_test_key"

    def test_get_binance_credentials_helper(self):
        """Test get_binance_credentials() Helper."""
        reset_secrets_manager()

        with patch.dict(
            os.environ,
            {"BINANCE_API_KEY": "helper_key", "BINANCE_API_SECRET": "helper_secret"},
        ):
            key, secret = get_binance_credentials()
            assert key == "helper_key"
            assert secret == "helper_secret"

    def test_get_telegram_credentials_helper(self):
        """Test get_telegram_credentials() Helper."""
        reset_secrets_manager()

        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "bot_xyz", "TELEGRAM_CHAT_ID": "chat_789"},
        ):
            token, chat_id = get_telegram_credentials()
            assert token == "bot_xyz"
            assert chat_id == "chat_789"

    def test_get_secret_key_helper(self):
        """Test get_secret_key() Helper."""
        reset_secrets_manager()

        with patch.dict(os.environ, {"SECRET_KEY": "jwt-secret-key"}):
            key = get_secret_key()
            assert key == "jwt-secret-key"

    def test_get_admin_credentials_helper(self):
        """Test get_admin_credentials() Helper."""
        reset_secrets_manager()

        with patch.dict(
            os.environ,
            {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "secure123"},
        ):
            username, password = get_admin_credentials()
            assert username == "admin"
            assert password == "secure123"

    def test_get_kucoin_credentials_helper(self):
        """Test get_kucoin_credentials() Helper."""
        reset_secrets_manager()

        with patch.dict(
            os.environ,
            {
                "KUCOIN_API_KEY": "kucoin_key",
                "KUCOIN_API_SECRET": "kucoin_secret",
                "KUCOIN_PASSPHRASE": "kucoin_pass",
            },
        ):
            key, secret, passphrase = get_kucoin_credentials()
            assert key == "kucoin_key"
            assert secret == "kucoin_secret"
            assert passphrase == "kucoin_pass"

    def test_get_bybit_credentials_helper(self):
        """Test get_bybit_credentials() Helper."""
        reset_secrets_manager()

        with patch.dict(
            os.environ,
            {"BYBIT_API_KEY": "bybit_key", "BYBIT_API_SECRET": "bybit_secret"},
        ):
            key, secret = get_bybit_credentials()
            assert key == "bybit_key"
            assert secret == "bybit_secret"

    def test_get_github_credentials_helper(self):
        """Test get_github_credentials() Helper."""
        reset_secrets_manager()

        with patch.dict(
            os.environ,
            {"GITHUB_TOKEN": "ghp_token", "GITHUB_USER": "myuser"},
        ):
            token, username = get_github_credentials()
            assert token == "ghp_token"
            assert username == "myuser"

    def test_get_ably_key_helper(self):
        """Test get_ably_key() Helper."""
        reset_secrets_manager()

        with patch.dict(os.environ, {"ABLY_API_KEY": "ably_key_123"}):
            key = get_ably_key()
            assert key == "ably_key_123"

    def test_get_colab_token_helper(self):
        """Test get_colab_token() Helper."""
        reset_secrets_manager()

        with patch.dict(os.environ, {"BT4T_API_TOKEN": "colab_token_456"}):
            token = get_colab_token()
            assert token == "colab_token_456"


class TestSecretMetadata:
    """Tests für SecretMetadata."""

    def test_secret_metadata_creation(self):
        """Test SecretMetadata Erstellung."""
        metadata = SecretMetadata(
            key="binance/api_key",
            version=2,
            created_at="2026-01-01T00:00:00Z",
            expires_at="2027-01-01T00:00:00Z",
            rotation_policy="30days",
        )

        assert metadata.key == "binance/api_key"
        assert metadata.version == 2
        assert metadata.rotation_policy == "30days"


class TestEdgeCases:
    """Edge Case Tests."""

    def test_empty_string_in_env(self):
        """Test dass leere Strings in ENV als leer behandelt werden."""
        with patch.dict(os.environ, {"BINANCE_API_KEY": ""}):
            manager = EnvironmentSecretsManager()
            # Leere Strings sollten als solche zurückgegeben werden
            # nicht als None
            result = manager.get("binance/api_key", default="default")
            assert result == ""

    def test_nonexistent_key_without_default(self):
        """Test nicht existierender Key ohne Default."""
        manager = EnvironmentSecretsManager()
        result = manager.get("completely/nonexistent")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
