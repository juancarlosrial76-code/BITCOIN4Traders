"""
Secrets Management Module - SOTA Implementation
==============================================

Secure API key and secret management with HashiCorp Vault support.

Security Features:
- Encrypted storage (Vault)
- Audit logging (all access attempts)
- Secret rotation support
- No secrets in environment variables (injection at runtime)
- Graceful degradation (Vault unavailable → fallback)

Priority Order:
1. HashiCorp Vault (highest security)
2. AWS Secrets Manager (optional)
3. Environment Variables (fallback)
4. .env file (development only)

Usage:
    # With Vault
    from src.config.secrets_manager import VaultSecretsManager

    secrets = VaultSecretsManager(vault_url="http://localhost:8200")
    api_key = secrets.get("binance/api_key")

    # With AWS
    from src.config.secrets_manager import AWSSecretsManager

    secrets = AWSSecretsManager(region="eu-central-1")
    api_key = secrets.get("binance/api_key")

Quick Start:
    from src.config import get_secrets, get_binance_credentials

    # Auto-detect backend (Vault → AWS → Environment)
    secrets = get_secrets()

    # Or direct helper functions
    api_key, api_secret = get_binance_credentials()

Configuration:
    Environment Variables:
        - VAULT_ADDR: Vault server URL (default: http://localhost:8200)
        - VAULT_TOKEN: Vault authentication token
        - VAULT_TOKEN_PATH: Path to vault token file (default: ~/.vault-token)
        - AWS_REGION: AWS region for Secrets Manager (default: eu-central-1)
        - BINANCE_API_KEY: Binance API key
        - BINANCE_API_SECRET: Binance API secret
        - TELEGRAM_BOT_TOKEN: Telegram bot token
        - TELEGRAM_CHAT_ID: Telegram chat ID
        - GITHUB_TOKEN: GitHub token
        - OPENROUTER_API_KEY: OpenRouter API key

Vault Setup (for production):
    1. Install Vault: https://www.vaultproject.io/downloads
    2. Start Vault server: vault server -dev
    3. Export token: export VAULT_ADDR=http://localhost:8200
    4. Export token: export VAULT_TOKEN=<your-token>
    5. Write secrets:
       vault kv put secret/bitcoin4traders/binance/api_key value=your_api_key
       vault kv put secret/bitcoin4traders/binance/api_secret value=your_api_secret

Development (.env file):
    The .env file is used as fallback when Vault/AWS is unavailable.
    It is NOT tracked in git (.gitignore).
    To create: cp .env.example .env

Testing:
    pytest tests/test_secrets_manager.py -v
"""

import os
import time
import logging
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager


_global_manager = None
_manager_lock = threading.Lock()

try:
    import hvac
except ImportError:
    hvac = None

try:
    import boto3
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""

    value: str
    timestamp: float
    ttl: int = 300  # 5 minutes default


class SecretNotFoundError(Exception):
    """Raised when a secret is not found."""

    pass


class VaultUnavailableError(Exception):
    """Raised when Vault is not reachable."""

    pass


@dataclass
class SecretMetadata:
    """Metadata for a secret."""

    key: str
    version: int = 1
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    rotation_policy: Optional[str] = None


@dataclass
class Secrets:
    """Container for all secrets with metadata."""

    # Exchange APIs
    binance_api_key: str = ""
    binance_api_secret: str = ""
    kucoin_api_key: str = ""
    kucoin_api_secret: str = ""
    kucoin_passphrase: str = ""
    bybit_api_key: str = ""
    bybit_api_secret: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # GitHub
    github_token: Optional[str] = None
    github_username: Optional[str] = None

    # AI / OpenRouter
    openrouter_api_key: Optional[str] = None

    # Backend (Phase 1)
    backend_secret_key: str = ""
    backend_admin_username: str = "admin"
    backend_admin_password: str = ""

    # Colab / Ably (Phase 3)
    ably_api_key: str = ""
    colab_api_token: str = ""
    colab_control_token: str = ""

    _metadata: Dict[str, SecretMetadata] = field(default_factory=dict)

    def is_binance_configured(self) -> bool:
        """Check if Binance API keys are configured."""
        return bool(self.binance_api_key and self.binance_api_secret)

    def is_kucoin_configured(self) -> bool:
        """Check if KuCoin API keys are configured."""
        return bool(
            self.kucoin_api_key and self.kucoin_api_secret and self.kucoin_passphrase
        )

    def is_bybit_configured(self) -> bool:
        """Check if Bybit API keys are configured."""
        return bool(self.bybit_api_key and self.bybit_api_secret)

    def is_telegram_configured(self) -> bool:
        """Check if Telegram credentials are configured."""
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def is_backend_configured(self) -> bool:
        """Check if backend credentials are configured."""
        return bool(self.backend_secret_key and self.backend_admin_password)

    def mask_sensitive(self) -> Dict[str, Any]:
        """Return dictionary with masked values."""

        def mask(value: str, show_last: int = 4) -> str:
            if not value:
                return ""
            if len(value) <= show_last:
                return "*" * len(value)
            return "*" * (len(value) - show_last) + value[-show_last:]

        return {
            "binance_api_key": mask(self.binance_api_key),
            "binance_api_secret": mask(self.binance_api_secret),
            "telegram_bot_token": mask(self.telegram_bot_token),
            "telegram_chat_id": self.telegram_chat_id,
            "github_token": mask(self.github_token or ""),
        }


class BaseSecretsManager(ABC):
    """Abstract base class for secrets managers."""

    ENV_MAPPING: Dict[str, str] = {
        # Binance
        "binance/api_key": "BINANCE_API_KEY",
        "binance/api_secret": "BINANCE_API_SECRET",
        # Telegram
        "telegram/bot_token": "TELEGRAM_BOT_TOKEN",
        "telegram/chat_id": "TELEGRAM_CHAT_ID",
        # GitHub
        "github/token": "GITHUB_TOKEN",
        "github/username": "GITHUB_USER",
        # OpenRouter (AI)
        "openrouter/api_key": "OPENROUTER_API_KEY",
        # Backend (Phase 1)
        "backend/secret_key": "SECRET_KEY",
        "backend/admin_password": "ADMIN_PASSWORD",
        "backend/admin_username": "ADMIN_USERNAME",
        # Multi-Exchange (Phase 2)
        "exchange/kucoin_key": "KUCOIN_API_KEY",
        "exchange/kucoin_secret": "KUCOIN_API_SECRET",
        "exchange/kucoin_passphrase": "KUCOIN_PASSPHRASE",
        "exchange/bybit_key": "BYBIT_API_KEY",
        "exchange/bybit_secret": "BYBIT_API_SECRET",
        # Colab (Phase 3)
        "ably/api_key": "ABLY_API_KEY",
        "colab/api_token": "BT4T_API_TOKEN",
        "colab/control_token": "CONTROL_API_TOKEN",
    }

    def __init__(
        self,
        mount_point: str = "secret",
        path_prefix: str = "bitcoin4traders",
        required_secrets: Optional[list] = None,
        audit_enabled: bool = True,
        cache_ttl: int = 300,
    ):
        """
        Args:
            mount_point: Vault mount point (default: secret)
            path_prefix: Path prefix for all secrets
            required_secrets: List of required secret paths
            audit_enabled: Whether to log access attempts
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 min)
        """
        self._mount_point = mount_point
        self._path_prefix = path_prefix
        self._required_secrets = required_secrets or []
        self._audit_enabled = audit_enabled
        self._cache_ttl = cache_ttl
        self._secrets_cache: Dict[str, CacheEntry] = {}
        self._initialized = False

    @abstractmethod
    def _load_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Load a secret from the backend."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret by key.

        Args:
            key: Secret key (e.g., "binance/api_key")
            default: Fallback value

        Returns:
            Secret value or default
        """
        # Check cache with TTL
        if key in self._secrets_cache:
            entry = self._secrets_cache[key]
            if time.time() - entry.timestamp < self._cache_ttl:
                self._audit_access(key, "cache_hit")
                return entry.value
            else:
                # Cache expired
                del self._secrets_cache[key]
                self._audit_access(key, "cache_expired")

        env_var = self.ENV_MAPPING.get(key)

        full_path = f"{self._path_prefix}/{key}"
        value = None

        try:
            secret_data = self._load_secret(full_path)
            if secret_data:
                value = secret_data.get("value") or secret_data.get(key.split("/")[-1])
                self._audit_access(key, "success")
        except Exception as e:
            self._audit_access(key, f"error: {e}")
            logger.warning(f"Failed to load secret {key}: {e}")

        if value is None and env_var:
            value = os.getenv(env_var, default)
            self._audit_access(key, "fallback")
        elif value is None:
            self._audit_access(key, "not_found")
            return default

        if value is not None:
            self._secrets_cache[key] = CacheEntry(
                value=value, timestamp=time.time(), ttl=self._cache_ttl
            )

        return value if value is not None else default

    def _audit_access(self, key: str, status: str) -> None:
        """Log secret access (Security Audit)."""
        if not self._audit_enabled:
            return

        logger.info(f"[AUDIT] Secret access: {key} - {status}")

    def get_all(self) -> Secrets:
        """Load all configured secrets."""
        return Secrets(
            # Exchange APIs
            binance_api_key=self.get("binance/api_key", "") or "",
            binance_api_secret=self.get("binance/api_secret", "") or "",
            kucoin_api_key=self.get("exchange/kucoin_key", "") or "",
            kucoin_api_secret=self.get("exchange/kucoin_secret", "") or "",
            kucoin_passphrase=self.get("exchange/kucoin_passphrase", "") or "",
            bybit_api_key=self.get("exchange/bybit_key", "") or "",
            bybit_api_secret=self.get("exchange/bybit_secret", "") or "",
            # Telegram
            telegram_bot_token=self.get("telegram/bot_token", "") or "",
            telegram_chat_id=self.get("telegram/chat_id", "") or "",
            # GitHub
            github_token=self.get("github/token"),
            github_username=self.get("github/username"),
            # OpenRouter
            openrouter_api_key=self.get("openrouter/api_key"),
            # Backend
            backend_secret_key=self.get("backend/secret_key", "") or "",
            backend_admin_username=self.get("backend/admin_username", "admin")
            or "admin",
            backend_admin_password=self.get("backend/admin_password", "") or "",
            # Colab
            ably_api_key=self.get("ably/api_key", "") or "",
            colab_api_token=self.get("colab/api_token", "") or "",
            colab_control_token=self.get("colab/control_token", "") or "",
        )

    def get_binance_credentials(self) -> tuple[str, str]:
        """Get Binance credentials as tuple."""
        return (
            self.get("binance/api_key", "") or "",
            self.get("binance/api_secret", "") or "",
        )

    def reload(self) -> None:
        """Reload secrets (clear cache)."""
        self._secrets_cache.clear()
        self._initialized = False


class VaultSecretsManager(BaseSecretsManager):
    """
    HashiCorp Vault Integration.

    Features:
    - Dynamic secrets
    - Audit logging
    - Secret rotation
    - Encrypted at rest

    Usage:
        secrets = VaultSecretsManager(
            vault_url="http://localhost:8200",
            token_path="/path/to/.vault-token"
        )
        api_key = secrets.get("binance/api_key")
    """

    def __init__(
        self,
        vault_url: Optional[str] = None,
        token: Optional[str] = None,
        token_path: Optional[str] = None,
        mount_point: str = "secret",
        path_prefix: str = "bitcoin4traders",
        namespace: Optional[str] = None,
        ssl_verify: bool = True,
        timeout: int = 10,
        max_retries: int = 3,
        rate_limit: int = 100,
        rate_period: int = 60,
        **kwargs,
    ):
        """
        Args:
            vault_url: Vault server URL (also via VAULT_ADDR env)
            token: Vault token (also via VAULT_TOKEN env)
            token_path: Path to vault token file
            namespace: Vault namespace (Enterprise)
            ssl_verify: Verify SSL certificate
            timeout: Request timeout in seconds (default: 10)
            max_retries: Max retries on timeout (default: 3)
            rate_limit: Max requests per period (default: 100)
            rate_period: Rate limit period in seconds (default: 60)
        """
        super().__init__(mount_point=mount_point, path_prefix=path_prefix, **kwargs)

        self._vault_url = vault_url or os.getenv("VAULT_ADDR", "http://localhost:8200")
        self._token = token or os.getenv("VAULT_TOKEN")
        self._token_path = token_path or os.getenv("VAULT_TOKEN_PATH", "~/.vault-token")
        self._namespace = namespace or os.getenv("VAULT_NAMESPACE")
        self._ssl_verify = ssl_verify
        self._timeout = timeout
        self._max_retries = max_retries

        # Rate limiting
        self._rate_limit = rate_limit
        self._rate_period = rate_period
        self._rate_lock = threading.Lock()
        self._rate_requests: list = []

        self._client: Optional[Any] = None

        if hvac is None:
            logger.warning("hvac not installed. Run: pip install hvac")

        self._init_client()

    def _init_client(self) -> None:
        """Initialize Vault client."""
        if hvac is None:
            return

        if not self._token:
            self._token = self._load_token()

        if not self._token:
            logger.warning("No Vault token available")
            return

        try:
            self._client = hvac.Client(
                url=self._vault_url,
                token=self._token,
                namespace=self._namespace,
                verify=self._ssl_verify,
            )

            if self._client.is_authenticated():
                self._initialized = True
                logger.info(f"Vault connected: {self._vault_url}")
            else:
                logger.warning("Vault authentication failed")
        except Exception as e:
            logger.warning(f"Vault initialization failed: {e}")

    def _load_token(self) -> Optional[str]:
        """Load token from file."""
        if self._token_path:
            token_path = Path(self._token_path).expanduser()
            if token_path.exists():
                return token_path.read_text().strip()
        return None

    def is_available(self) -> bool:
        """Check if Vault is reachable and authenticated."""
        if not self._client:
            return False

        try:
            return self._client.is_authenticated()
        except Exception:
            return False

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limit."""
        if self._rate_limit <= 0:
            return

        now = time.time()
        with self._rate_lock:
            # Remove old requests
            self._rate_requests = [
                t for t in self._rate_requests if now - t < self._rate_period
            ]

            if len(self._rate_requests) >= self._rate_limit:
                wait_time = self._rate_period - (now - self._rate_requests[0])
                if wait_time > 0:
                    logger.warning(
                        f"Vault rate limit reached, waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)

            self._rate_requests.append(now)

    def _load_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Load secret from Vault with timeout and retry."""
        if not self._client or not self.is_available():
            return None

        # Check rate limit
        self._check_rate_limit()

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.secrets.kv.v2.read_secret_version(
                    path=path, mount_point=self._mount_point
                )
                return response["data"]["data"]
            except hvac.exceptions.VaultNotFound:
                logger.debug(f"Secret not found in Vault: {path}")
                return None
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"Vault read error (attempt {attempt + 1}/{self._max_retries}): {e}"
                    )
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(
                        f"Vault read failed after {self._max_retries} attempts: {e}"
                    )

        return None

    def write_secret(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Write a secret to Vault.

        Args:
            key: Secret key
            value: Dictionary with secret data

        Returns:
            True if successful
        """
        if not self._client or not self.is_available():
            logger.error("Vault not available for write")
            return False

        full_path = f"{self._path_prefix}/{key}"

        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=full_path, secret={**value}, mount_point=self._mount_point
            )
            self._secrets_cache.clear()
            logger.info(f"Secret written to Vault: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to write secret {key}: {e}")
            return False

    def delete_secret(self, key: str) -> bool:
        """Delete a secret from Vault."""
        if not self._client or not self.is_available():
            return False

        full_path = f"{self._path_prefix}/{key}"

        try:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=full_path, mount_point=self._mount_point
            )
            self._secrets_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {e}")
            return False


class AWSSecretsManager(BaseSecretsManager):
    """
    AWS Secrets Manager Integration.

    Features:
    - Integrated with AWS IAM
    - KMS encryption
    - Automatic rotation support
    - CloudTrail audit

    Usage:
        secrets = AWSSecretsManager(region="eu-central-1")
        api_key = secrets.get("binance/api_key")
    """

    def __init__(
        self,
        region: Optional[str] = None,
        secret_prefix: str = "bitcoin4traders",
        **kwargs,
    ):
        super().__init__(path_prefix=secret_prefix, **kwargs)

        self._region = region or os.getenv("AWS_REGION", "eu-central-1")
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if boto3 is None:
            logger.warning("boto3 not installed. Run: pip install boto3")
            return

        try:
            self._client = boto3.client("secretsmanager", region_name=self._region)
            self._initialized = True
        except Exception as e:
            logger.warning(f"AWS client init failed: {e}")

    def is_available(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.describe_secret(SecretId="test")
            return True
        except Exception:
            return False

    def _load_secret(self, path: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None

        try:
            response = self._client.get_secret_value(SecretId=path)
            secret_string = response["SecretString"]
            return json.loads(secret_string)
        except self._client.exceptions.ResourceNotFoundException:
            return None
        except Exception as e:
            logger.warning(f"AWS secrets manager error for {path}: {e}")
            return None


class EnvironmentSecretsManager(BaseSecretsManager):
    """
    Fallback: Environment Variables only.

    Uses only environment variables - no external storage.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = True

    def is_available(self) -> bool:
        return True

    def _load_secret(self, path: str) -> Optional[Dict[str, Any]]:
        key = path.replace("/", "_").upper()
        value = os.getenv(key)

        if value:
            return {"value": value}
        return None


@contextmanager
def secrets_manager_context(backend: str = "auto", **kwargs):
    """
    Context manager for secrets manager.

    Usage:
        with secrets_manager_context("vault", vault_url="...") as sm:
            api_key = sm.get("binance/api_key")
    """
    manager = create_secrets_manager(backend, **kwargs)
    try:
        yield manager
    finally:
        pass


def create_secrets_manager(backend: str = "auto", **kwargs) -> BaseSecretsManager:
    """
    Factory function to create secrets manager.

    Args:
        backend: "vault", "aws", "env" or "auto" (try vault → aws → env)

    Returns:
        SecretsManager instance
    """
    if backend == "auto":
        if hvac:
            vault_mgr = VaultSecretsManager(**kwargs)
            if vault_mgr.is_available():
                return vault_mgr

        if boto3:
            aws_mgr = AWSSecretsManager(**kwargs)
            if aws_mgr.is_available():
                return aws_mgr

        return EnvironmentSecretsManager(**kwargs)

    if backend == "vault":
        return VaultSecretsManager(**kwargs)

    if backend == "aws":
        return AWSSecretsManager(**kwargs)

    if backend == "env":
        return EnvironmentSecretsManager(**kwargs)

    raise ValueError(f"Unknown backend: {backend}")


def get_secrets(backend: str = "auto", auto_backup: bool = True, **kwargs) -> Secrets:
    """
    Global function to get all secrets.

    Args:
        backend: "vault", "aws", "env" or "auto"
        auto_backup: Whether to create automatic backup if .env changed

    Returns:
        Secrets dataclass
    """
    global _global_manager

    # Extract auto_backup from kwargs if present (prevent passing to manager)
    _auto_backup = kwargs.pop("auto_backup", True)
    _auto_backup = auto_backup if auto_backup is not None else _auto_backup

    # Thread-safe initialization
    should_init = False
    with _manager_lock:
        if _global_manager is None:
            should_init = True
            _global_manager = create_secrets_manager(backend, **kwargs)

    # Auto-backup only on first call (outside lock to avoid blocking)
    if should_init and _auto_backup:
        try:
            from src.config.secure_backup import auto_backup as do_backup

            do_backup()
        except Exception:
            pass  # Silent fail - backup is optional

    return _global_manager.get_all()


def get_binance_credentials(
    backend: str = "auto", auto_backup: bool = False, **kwargs
) -> tuple[str, str]:
    """Get Binance credentials."""
    secrets = get_secrets(backend, auto_backup=auto_backup, **kwargs)
    return secrets.binance_api_key, secrets.binance_api_secret


def get_telegram_credentials(
    backend: str = "auto", auto_backup: bool = False, **kwargs
) -> tuple[str, str]:
    """Get Telegram credentials."""
    secrets = get_secrets(backend, auto_backup=auto_backup, **kwargs)
    return secrets.telegram_bot_token, secrets.telegram_chat_id


def reset_secrets_manager() -> None:
    """Reset the global manager."""
    global _global_manager
    with _manager_lock:
        _global_manager = None

    # Also reset AgeBackup hash
    try:
        from src.config.secure_backup import AgeBackup

        AgeBackup().reset_hash()
    except Exception:
        pass
