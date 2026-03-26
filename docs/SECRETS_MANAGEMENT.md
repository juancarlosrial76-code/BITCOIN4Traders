# Secrets Management

SOTA secrets management for BITCOIN4Traders.

## Overview

This module provides secure API key and secret management with support for multiple backends:

- **HashiCorp Vault** (recommended for production)
- **AWS Secrets Manager** (if using AWS)
- **Environment Variables** (fallback)
- **.env file** (development only)

## Priority Order

```
1. HashiCorp Vault (highest security)
   ├── Encrypted at rest
   ├── Audit logging
   └── Secret rotation

2. AWS Secrets Manager
   ├── KMS encryption
   ├── CloudTrail audit
   └── Automatic rotation

3. Environment Variables
   └── Fallback when Vault/AWS unavailable

4. .env file (development only)
```

## Features

- Encrypted storage (Vault/KMS)
- Audit logging (all access attempts)
- Secret rotation support
- Graceful degradation (Vault unavailable → fallback)
- Secret masking in logs
- **Cache TTL** - Configurable cache expiration (default: 5 minutes)
- **Vault Timeout** - Request timeout with retry logic (default: 10s, 3 retries)
- **Rate Limiting** - Prevents Vault rate limit errors (default: 100 req/60s)

## Quick Start

```python
from src.config import get_secrets, get_binance_credentials

# Auto-detect backend (Vault → AWS → Environment)
secrets = get_secrets()

# Or use helper functions
api_key, api_secret = get_binance_credentials()
```

## Configuration

### Environment Variables

All supported secrets:

| Variable             | Description                                       |
| -------------------- | ------------------------------------------------- |
| **Exchange APIs**    |                                                   |
| `BINANCE_API_KEY`    | Binance API key                                   |
| `BINANCE_API_SECRET` | Binance API secret                                |
| `KUCOIN_API_KEY`     | KuCoin API key                                    |
| `KUCOIN_API_SECRET`  | KuCoin API secret                                 |
| `KUCOIN_PASSPHRASE`  | KuCoin passphrase                                 |
| `BYBIT_API_KEY`      | Bybit API key                                     |
| `BYBIT_API_SECRET`   | Bybit API secret                                  |
| **Backend**          |                                                   |
| `SECRET_KEY`         | JWT secret for backend authentication             |
| `ADMIN_USERNAME`     | Backend admin username (default: admin)           |
| `ADMIN_PASSWORD`     | Backend admin password                            |
| **Telegram**         |                                                   |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token                                |
| `TELEGRAM_CHAT_ID`   | Telegram chat ID                                  |
| **GitHub**           |                                                   |
| `GITHUB_TOKEN`       | GitHub API token                                  |
| `GITHUB_USER`        | GitHub username                                   |
| **AI**               |                                                   |
| `OPENROUTER_API_KEY` | OpenRouter API key                                |
| **Colab**            |                                                   |
| `ABLY_API_KEY`       | Ably API key (for real-time communication)        |
| `BT4T_API_TOKEN`     | Colab extension API token                         |
| `CONTROL_API_TOKEN`  | Control plane API token                           |
| **Vault/AWS**        |                                                   |
| `VAULT_ADDR`         | Vault server URL (default: http://localhost:8200) |
| `VAULT_TOKEN`        | Vault authentication token                        |
| `AWS_REGION`         | AWS region (default: eu-central-1)                |

## Advanced Configuration

### Cache TTL

```python
# Default: 300 seconds (5 minutes)
secrets = get_secrets(cache_ttl=600)  # 10 minutes
```

### Vault Timeout & Retry

```python
from src.config.secrets_manager import VaultSecretsManager

vault = VaultSecretsManager(
    vault_url="http://localhost:8200",
    timeout=30,           # 30 second timeout
    max_retries=5,       # 5 retries on timeout
    rate_limit=50,       # 50 requests per period
    rate_period=60       # 60 second period
)
```

### Helper Functions

```python
# Import all helpers
from src.config import (
    get_secrets,
    get_binance_credentials,
    get_telegram_credentials,
    get_secret_key,
    get_admin_credentials,
    get_kucoin_credentials,
    get_bybit_credentials,
    get_github_credentials,
    get_ably_key,
    get_colab_token,
    get_control_token,
)

# Usage examples
api_key, api_secret = get_binance_credentials()
telegram_token, chat_id = get_telegram_credentials()
secret_key = get_secret_key()
username, password = get_admin_credentials()
kucoin_key, kucoin_secret, passphrase = get_kucoin_credentials()
bybit_key, bybit_secret = get_bybit_credentials()
token, user = get_github_credentials()
ably_key = get_ably_key()
colab_token = get_colab_token()
control_token = get_control_token()
```

## .env File (Development)

The `.env` file is used as fallback when Vault/AWS is unavailable.

```
# Create from template
cp .env.example .env

# Edit with your secrets
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
SECRET_KEY=your_jwt_secret
ADMIN_PASSWORD=your_secure_password
```

Note: `.env` is ignored by git (see `.gitignore`).

## Vault Setup (Production)

### 1. Install Vault

```bash
# Download from https://www.vaultproject.io/downloads
# or use Homebrew
brew install vault
```

### 2. Start Vault Server

```bash
# Development mode (in-memory)
vault server -dev

# Export the address
export VAULT_ADDR='http://127.0.0.1:8200'

# Export the token (shown in server output)
export VAULT_TOKEN='your-dev-token'
```

### 3. Write Secrets

```bash
# Write Binance credentials
vault kv put secret/bitcoin4traders/binance/api_key value=your_actual_api_key
vault kv put secret/bitcoin4traders/binance/api_secret value=your_actual_api_secret

# Write Backend credentials
vault kv put secret/bitcoin4traders/backend/secret_key value=your_jwt_secret
vault kv put secret/bitcoin4traders/backend/admin_password value=your_secure_password

# Write Telegram credentials (optional)
vault kv put secret/bitcoin4traders/telegram/bot_token value=your_bot_token
vault kv put secret/bitcoin4traders/telegram/chat_id value=your_chat_id
```

### 4. Verify

```bash
# Test reading
vault kv get secret/bitcoin4traders/binance/api_key
```

## Security Best Practices

1. **Production**: Use HashiCorp Vault
2. **Never commit secrets to git** (`.env` is gitignored)
3. **Rotate secrets regularly**
4. **Enable audit logging** (enabled by default)
5. **Use least-privilege access** (Vault policies)

## Testing

```bash
# Run tests
pytest tests/test_secrets_manager.py -v
```

All 28 tests must pass.

## Migration from Environment Variables

If you previously used environment variables directly:

```python
# OLD (deprecated)
import os
api_key = os.getenv("BINANCE_API_KEY")

# NEW (recommended)
from src.config import get_binance_credentials
api_key, api_secret = get_binance_credentials()
```

The new approach:

- Works with Vault, AWS, or environment variables
- Provides audit logging
- Supports secret masking in logs
- Graceful fallback when primary backend unavailable
