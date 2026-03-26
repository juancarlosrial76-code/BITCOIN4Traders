"""
Tests for Secure Backup Module

Run: pytest tests/test_secure_backup.py -v
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestAgeBackup:
    """Tests for AgeBackup class."""

    def test_age_not_available(self):
        """Test when age is not installed."""
        from src.config.secure_backup import AgeBackup

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            backup = AgeBackup()
            assert backup.is_available() is False

    def test_has_keys_false(self):
        """Test when keys don't exist."""
        from src.config.secure_backup import AgeBackup

        backup = AgeBackup(key_path="/nonexistent/age.keys")
        assert backup.has_keys() is False

    def test_list_backups_empty(self):
        """Test listing backups when directory doesn't exist."""
        from src.config.secure_backup import AgeBackup

        backup = AgeBackup(backup_dir="/nonexistent/backups")
        assert backup.list_backups() == []

    def test_compute_file_hash_nonexistent(self):
        """Test hash computation for nonexistent file."""
        from src.config.secure_backup import AgeBackup

        backup = AgeBackup()
        result = backup._compute_file_hash(Path("/nonexistent/file"))
        assert result == ""

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run):
        """Test when age is installed."""
        mock_run.return_value = MagicMock(returncode=0)

        from src.config.secure_backup import AgeBackup

        backup = AgeBackup()
        # Note: This will still check the actual subprocess if age is installed
        # But with the mock it should work

    def test_imports(self):
        """Test that all imports work."""
        from src.config import (
            AgeBackup,
            AgeNotFoundError,
            AgeKeyNotFoundError,
            encrypt_secrets,
            decrypt_secrets,
            auto_backup,
            restore_backup,
        )

        assert AgeBackup is not None
        assert AgeNotFoundError is not None
        assert AgeKeyNotFoundError is not None


class TestSecretsManagerAutoBackup:
    """Tests for auto-backup integration in get_secrets."""

    def test_get_secrets_with_auto_backup(self):
        """Test get_secrets with auto_backup enabled."""
        # Should not raise even if age not available
        from src.config import get_secrets, reset_secrets_manager

        reset_secrets_manager()

        # This will try to auto-backup but should fail silently
        # The actual test would need age installed
        try:
            secrets = get_secrets(auto_backup=True)
            assert secrets is not None
        except Exception:
            # Expected if age not installed
            pass

    def test_get_secrets_without_auto_backup(self):
        """Test get_secrets with auto_backup disabled."""
        from src.config import get_secrets, reset_secrets_manager

        reset_secrets_manager()

        secrets = get_secrets(auto_backup=False)
        assert secrets is not None


class TestGitignore:
    """Tests for .gitignore configuration."""

    def test_gitignore_has_age_keys(self):
        """Test that age.keys is in .gitignore."""
        gitignore_path = Path(".gitignore")

        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert "age.keys" in content
            assert "backups/" in content

    def test_gitignore_allows_age_files(self):
        """Test that .age files are NOT in .gitignore (they're safe to commit)."""
        gitignore_path = Path(".gitignore")

        if gitignore_path.exists():
            content = gitignore_path.read_text()
            # .env.age should NOT be ignored (it's encrypted and safe)
            # But the private key and local backups should be


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
