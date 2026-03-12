#!/bin/bash

# BITCOIN4Traders - Complete Installation Script
# This script sets up the entire trading system locally

set -e

echo "=========================================="
echo "BITCOIN4Traders Installation"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.11"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.11+ is required. Current version: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}Python version OK: $python_version${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies for frontend
echo -e "${YELLOW}Setting up frontend...${NC}"
cd frontend
npm install
cd ..

# Copy environment file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << 'EOF'
# Server Configuration
SECRET_KEY=your-secret-key-change-in-production
ADMIN_PASSWORD=admin123

# Binance API (optional - for live trading)
# BINANCE_API_KEY=your_api_key
# BINANCE_API_SECRET=your_api_secret
# BINANCE_TESTNET=true

# Colab Bridge (optional)
# BT4T_LISTENER_URL=https://your-url.trycloudflare.com
# BT4T_API_TOKEN=bt4t-secret-token

# Server
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
ENVIRONMENT=development
EOF
fi

echo -e "${GREEN}=========================================="
echo "Installation Complete!"
echo -e "==========================================${NC}"
echo ""
echo "To start the application:"
echo ""
echo "1. Start Backend:"
echo "   source venv/bin/activate"
echo "   python -m uvicorn backend.main:app --reload"
echo ""
echo "2. Start Frontend (in another terminal):"
echo "   cd frontend && npm run dev"
echo ""
echo "3. Open your browser:"
echo "   http://localhost:5173"
echo ""
echo "Login credentials:"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "API Documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "HTML Manual:"
echo "   Open frontend/public/manual/index.html in browser"
