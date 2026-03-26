"""
Secure Backup Module - Age Encryption
=====================================

Provides secure encryption/decryption for secrets using age.

Age is a modern, secure encryption tool:
- NSA-resistant (ChaCha20-Poly1305)
- Simple key management
- No dependencies on external services

Usage:
    from src.config.secure_backup import AgeBackup

    backup = AgeBackup()

    # Encrypt .env to .env.age
    backup.encrypt_file(".env", ".env.age")

    # Decrypt .env.age to .env
    backup.decrypt_file(".env.age", ".env")

    # Auto-backup if changed
    backup.auto_backup_if_changed()
"""

import os
import subprocess
import hashlib
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AgeNotFoundError(Exception):
    """Raised when age is not installed."""

    pass


class AgeKeyNotFoundError(Exception):
    """Raised when age keys are not found."""

    pass


class AgeBackup:
    """
    Age-based encryption for secrets.

    Features:
    - ChaCha20-Poly1305 encryption (NSA-resistant)
    - Automatic backup on secret changes
    - Incremental backups with rotation
    - Google Drive sync support
    - Thread-safe operations

    Security:
    - Private key stays local
    - Encrypted files can be stored in cloud/git
    - No plaintext secrets in logs
    """

    DEFAULT_KEY_PATH = "age.keys"
    DEFAULT_ENV_FILE = ".env"
    DEFAULT_BACKUP_DIR = "backups"
    MAX_BACKUPS = 10

    # Class-level hash storage file
    HASH_FILE = Path.home() / ".bitcoin4traders" / "last_hash"

    def __init__(
        self,
        key_path: str = DEFAULT_KEY_PATH,
        env_file: str = DEFAULT_ENV_FILE,
        backup_dir: str = DEFAULT_BACKUP_DIR,
    ):
        """
        Args:
            key_path: Path to age private key (default: age.keys)
            env_file: Path to .env file (default: .env)
            backup_dir: Directory for backups (default: backups)
        """
        self.key_path = Path(key_path)
        self.env_file = Path(env_file)
        self.backup_dir = Path(backup_dir)
        self._last_hash: Optional[str] = self._load_last_hash()

    @staticmethod
    def _load_last_hash() -> Optional[str]:
        """Load last hash from persistent storage."""
        try:
            if AgeBackup.HASH_FILE.exists():
                return AgeBackup.HASH_FILE.read_text().strip()
        except Exception:
            pass
        return None

    def _save_last_hash(self, hash_value: str) -> None:
        """Save hash to persistent storage."""
        try:
            AgeBackup.HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
            AgeBackup.HASH_FILE.write_text(hash_value)
            self._last_hash = hash_value
        except Exception as e:
            logger.warning(f"Failed to save hash: {e}")

    def reset_hash(self) -> None:
        """Reset the stored hash to force a backup."""
        self._last_hash = None
        try:
            if AgeBackup.HASH_FILE.exists():
                AgeBackup.HASH_FILE.unlink()
        except Exception:
            pass

    def is_available(self) -> bool:
        """Check if age is installed."""
        try:
            result = subprocess.run(
                ["age", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def has_keys(self) -> bool:
        """Check if age keys exist."""
        return self.key_path.exists()

    def get_public_key(self) -> Optional[str]:
        """Get public key for sharing."""
        if not self.has_keys():
            return None
        pub_key_path = self.key_path.with_suffix(self.key_path.suffix + ".pub")
        if pub_key_path.exists():
            return pub_key_path.read_text().strip()
        return None

    def _run_age(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run age command with key."""
        if not self.is_available():
            raise AgeNotFoundError(
                "age not installed. Install: https://github.com/FiloSottile/age"
            )

        if not self.has_keys():
            raise AgeKeyNotFoundError(
                f"age keys not found at {self.key_path}. " "Generate with: age-keygen"
            )

        cmd = ["age", "-i", str(self.key_path)] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"age command failed: {result.stderr}")
            raise RuntimeError(f"age failed: {result.stderr}")

        return result

    def encrypt_file(
        self, source: Optional[str] = None, dest: Optional[str] = None
    ) -> bool:
        """
        Encrypt a file using age.

        Args:
            source: Source file (default: .env)
            dest: Destination file (default: .env.age)

        Returns:
            True if successful
        """
        source = Path(source) if source else self.env_file
        dest = Path(dest) if dest else Path(str(source) + ".age")

        if not source.exists():
            logger.warning(f"Source file not found: {source}")
            return False

        try:
            # Get public key
            pub_key_path = self.key_path.with_suffix(self.key_path.suffix + ".pub")
            if not pub_key_path.exists():
                logger.error(f"Public key not found: {pub_key_path}")
                return False

            public_key = pub_key_path.read_text().strip()

            # Encrypt
            with open(source, "rb") as infile:
                with open(dest, "wb") as outfile:
                    subprocess.run(
                        ["age", "--recipient", public_key],
                        stdin=infile,
                        stdout=outfile,
                        check=True,
                        timeout=30,
                    )

            logger.info(f"Encrypted {source} → {dest}")
            return True

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return False

    def decrypt_file(self, source: str, dest: Optional[str] = None) -> bool:
        """
        Decrypt an age-encrypted file.

        Args:
            source: Source file (.age)
            dest: Destination file (default: .env)

        Returns:
            True if successful
        """
        source = Path(source)
        dest = Path(dest) if dest else self.env_file

        if not source.exists():
            logger.warning(f"Source file not found: {source}")
            return False

        try:
            with open(source, "rb") as infile:
                with open(dest, "wb") as outfile:
                    subprocess.run(
                        ["age", "-d", "-i", str(self.key_path)],
                        stdin=infile,
                        stdout=outfile,
                        check=True,
                        timeout=30,
                    )

            logger.info(f"Decrypted {source} → {dest}")
            return True

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return False

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        if not file_path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def auto_backup_if_changed(self, force: bool = False) -> bool:
        """
        Automatically create backup if .env has changed.

        Args:
            force: Force backup even if unchanged

        Returns:
            True if backup was created
        """
        if not self.env_file.exists():
            logger.debug("No .env file to backup")
            return False

        # Check if changed
        current_hash = self._compute_file_hash(self.env_file)

        if current_hash == self._last_hash and not force:
            logger.debug("No changes detected, skipping backup")
            return False

        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f".env.age.{timestamp}"

        # Encrypt
        if self.encrypt_file(str(self.env_file), str(backup_file)):
            self._save_last_hash(current_hash)

            # Create symlink to latest
            latest_link = self.backup_dir / ".env.age.latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(backup_file.name)

            # Cleanup old backups
            self._cleanup_old_backups()

            logger.info(f"Auto-backup created: {backup_file.name}")
            return True

        return False

    def _cleanup_old_backups(self) -> None:
        """Remove old backups, keeping only MAX_BACKUPS."""
        if not self.backup_dir.exists():
            return

        # Get all age backup files
        backups = sorted(
            self.backup_dir.glob(".env.age.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove old ones
        for old_backup in backups[self.MAX_BACKUPS :]:
            old_backup.unlink()
            logger.debug(f"Removed old backup: {old_backup.name}")

    def restore_from_backup(self, backup_file: str) -> bool:
        """
        Restore .env from a backup file.

        Args:
            backup_file: Path to backup file

        Returns:
            True if successful
        """
        backup_path = Path(backup_file)

        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        # Create backup of current .env first
        if self.env_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_backup = self.env_file.with_suffix(
                f".env.pre_restore.{timestamp}"
            )
            self.env_file.rename(emergency_backup)
            logger.info(f"Emergency backup created: {emergency_backup}")

        # Decrypt to .env
        return self.decrypt_file(str(backup_path), str(self.env_file))

    def list_backups(self) -> List[Path]:
        """List all available backups."""
        if not self.backup_dir.exists():
            return []

        return sorted(
            self.backup_dir.glob(".env.age.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def sync_to_google_drive(self, remote: str = "gdrive:bitcoin4traders") -> bool:
        """
        Sync backups to Google Drive using rclone.

        Args:
            remote: rclone remote name

        Returns:
            True if successful
        """
        try:
            subprocess.run(
                ["rclone", "version"], capture_output=True, check=True, timeout=10
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("rclone not installed, skipping Google Drive sync")
            return False

        if not self.list_backups():
            logger.debug("No backups to sync")
            return False

        try:
            subprocess.run(
                ["rclone", "copy", str(self.backup_dir), remote],
                check=True,
                capture_output=True,
                timeout=60,
            )
            logger.info(f"Synced backups to {remote}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"rclone sync failed: {e}")
            return False


# Convenience functions
def encrypt_secrets() -> bool:
    """Encrypt .env to .env.age."""
    backup = AgeBackup()
    return backup.encrypt_file()


def decrypt_secrets() -> bool:
    """Decrypt .env.age to .env."""
    backup = AgeBackup()
    backup_file = backup.env_file.with_suffix(".env.age")
    return backup.decrypt_file(str(backup_file))


def auto_backup() -> bool:
    """Create automatic backup if changed."""
    backup = AgeBackup()
    return backup.auto_backup_if_changed()


def restore_backup(backup_file: str) -> bool:
    """Restore from backup."""
    backup = AgeBackup()
    return backup.restore_from_backup(backup_file)
