"""
Config Module - SOTA Secrets Management

Provides multiple secret backends:
- VaultSecretsManager: HashiCorp Vault (recommended)
- AWSSecretsManager: AWS Secrets Manager
- EnvironmentSecretsManager: Environment Variables only

Usage:
    # Auto-detect (tries Vault → AWS → Env)
    from src.config import get_secrets

    # Explicit with Vault
    from src.config.secrets_manager import VaultSecretsManager, get_secrets

    secrets = get_secrets(backend="vault", vault_url="http://localhost:8200")
    api_key, api_secret = get_binance_credentials(backend="vault")
"""

from typing import Optional

from src.config.secure_backup import (
    AgeBackup,
    AgeNotFoundError,
    AgeKeyNotFoundError,
    encrypt_secrets,
    decrypt_secrets,
    auto_backup,
    restore_backup,
)

from src.config.secrets_manager import (
    BaseSecretsManager,
    VaultSecretsManager,
    AWSSecretsManager,
    EnvironmentSecretsManager,
    Secrets,
    SecretMetadata,
    SecretNotFoundError,
    VaultUnavailableError,
    create_secrets_manager,
    get_secrets,
    get_binance_credentials,
    get_telegram_credentials,
    reset_secrets_manager,
    secrets_manager_context,
)


# Additional helper functions
def get_secret_key() -> str:
    """Get JWT SECRET_KEY for backend authentication."""
    return get_secrets().backend_secret_key or ""


def get_admin_credentials() -> tuple[str, str]:
    """Get (username, password) for backend admin access."""
    secrets = get_secrets()
    return (
        secrets.backend_admin_username or "admin",
        secrets.backend_admin_password or "",
    )


def get_kucoin_credentials() -> tuple[str, str, str]:
    """Get (api_key, api_secret, passphrase) for KuCoin."""
    secrets = get_secrets()
    return (
        secrets.kucoin_api_key or "",
        secrets.kucoin_api_secret or "",
        secrets.kucoin_passphrase or "",
    )


def get_bybit_credentials() -> tuple[str, str]:
    """Get (api_key, api_secret) for Bybit."""
    secrets = get_secrets()
    return (secrets.bybit_api_key or "", secrets.bybit_api_secret or "")


def get_github_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get (token, username) for GitHub."""
    secrets = get_secrets()
    return (secrets.github_token, secrets.github_username)


def get_ably_key() -> str:
    """Get Ably API key."""
    return get_secrets().ably_api_key or ""


def get_colab_token() -> str:
    """Get Colab API token."""
    return get_secrets().colab_api_token or ""


def get_control_token() -> str:
    """Get Control plane API token."""
    return get_secrets().colab_control_token or ""


__all__ = [
    # Secure Backup
    "AgeBackup",
    "AgeNotFoundError",
    "AgeKeyNotFoundError",
    "encrypt_secrets",
    "decrypt_secrets",
    "auto_backup",
    "restore_backup",
    # Base classes
    "BaseSecretsManager",
    "VaultSecretsManager",
    "AWSSecretsManager",
    "EnvironmentSecretsManager",
    # Data classes
    "Secrets",
    "SecretMetadata",
    # Exceptions
    "SecretNotFoundError",
    "VaultUnavailableError",
    # Factory functions
    "create_secrets_manager",
    "get_secrets",
    # Credential helpers
    "get_binance_credentials",
    "get_telegram_credentials",
    "get_secret_key",
    "get_admin_credentials",
    "get_kucoin_credentials",
    "get_bybit_credentials",
    "get_github_credentials",
    "get_ably_key",
    "get_colab_token",
    "get_control_token",
    # Utility
    "reset_secrets_manager",
    "secrets_manager_context",
]
