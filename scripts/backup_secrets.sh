#!/bin/bash
# =============================================================================
# Secure Secrets Backup Script
# =============================================================================
# Encrypts secrets with age and stores encrypted backup.
# 
# Prerequisites:
#   1. Install age: https://github.com/FiloSottile/age
#      macOS: brew install age
#      Linux: sudo apt install age or download from github
#
#   2. Generate keys (once):
#      age-keygen -o age.keys
#
#   3. Save age.keys to a secure location (USB, password manager, etc.)
#      NEVER commit age.keys to git!
#
# Usage:
#   ./scripts/backup_secrets.sh encrypt   # Encrypt .env → .env.age
#   ./scripts/backup_secrets.sh decrypt   # Decrypt .env.age → .env
#   ./scripts/backup_secrets.sh backup    # Encrypt + copy to backup folder
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGE_KEY_FILE="${SCRIPT_DIR}/../age.keys"
SOURCE_ENV="${SCRIPT_DIR}/../.env"
BACKUP_DIR="${SCRIPT_DIR}/../backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_dependencies() {
    if ! command -v age &> /dev/null; then
        log_error "age not installed!"
        echo "Install: https://github.com/FiloSottile/age"
        exit 1
    fi
    
    if [ ! -f "$AGE_KEY_FILE" ]; then
        log_error "age keys not found: $AGE_KEY_FILE"
        echo "Generate keys with: age-keygen -o age.keys"
        exit 1
    fi
}

generate_keys() {
    log_info "Generating age keys..."
    mkdir -p "$(dirname "$AGE_KEY_FILE")"
    age-keygen -o "$AGE_KEY_FILE"
    chmod 600 "$AGE_KEY_FILE"
    log_info "Keys saved to: $AGE_KEY_FILE"
    log_warn "IMPORTANT: Save this file to a secure location!"
}

encrypt() {
    check_dependencies
    
    if [ ! -f "$SOURCE_ENV" ]; then
        log_error "Source file not found: $SOURCE_ENV"
        exit 1
    fi
    
    local encrypted_file="${SOURCE_ENV}.age"
    
    log_info "Encrypting $SOURCE_ENV → $encrypted_file"
    age --recipient "$(cat "${AGE_KEY_FILE}.pub")" -o "$encrypted_file" "$SOURCE_ENV"
    
    log_info "Encrypted file created: $encrypted_file"
    echo ""
    echo "You can safely commit $encrypted_file to git or upload to cloud!"
    echo "Only someone with age.keys can decrypt it."
}

decrypt() {
    check_dependencies
    
    local encrypted_file="${SOURCE_ENV}.age"
    
    if [ ! -f "$encrypted_file" ]; then
        log_error "Encrypted file not found: $encrypted_file"
        exit 1
    fi
    
    log_info "Decrypting $encrypted_file → $SOURCE_ENV"
    age -d -i "$AGE_KEY_FILE" "$encrypted_file" > "$SOURCE_ENV"
    
    log_info "Decrypted to: $SOURCE_ENV"
}

backup() {
    check_dependencies
    
    mkdir -p "$BACKUP_DIR"
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${BACKUP_DIR}/.env.age.${timestamp}"
    
    log_info "Creating encrypted backup..."
    age --recipient "$(cat "${AGE_KEY_FILE}.pub")" -o "$backup_file" "$SOURCE_ENV"
    
    log_info "Backup saved to: $backup_file"
    
    # Keep only last 5 backups
    ls -1t "${BACKUP_DIR}"/.env.age.* 2>/dev/null | tail -n +6 | xargs -r rm
    
    log_info "Backup complete!"
}

show_public_key() {
    check_dependencies
    log_info "Public key:"
    cat "${AGE_KEY_FILE}.pub"
}

case "${1:-}" in
    generate-keys)
        generate_keys
        ;;
    encrypt)
        encrypt
        ;;
    decrypt)
        decrypt
        ;;
    backup)
        backup
        ;;
    show-key)
        show_public_key
        ;;
    *)
        echo "Usage: $0 {generate-keys|encrypt|decrypt|backup|show-key}"
        echo ""
        echo "Commands:"
        echo "  generate-keys  - Generate new age encryption keys (run once)"
        echo "  encrypt        - Encrypt .env → .env.age"
        echo "  decrypt        - Decrypt .env.age → .env"
        echo "  backup         - Create encrypted backup with timestamp"
        echo "  show-key       - Display public key for sharing"
        exit 1
        ;;
esac
