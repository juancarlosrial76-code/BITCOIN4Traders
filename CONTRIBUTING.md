# Contributing to BITCOIN4Traders

Thank you for your interest in contributing to BITCOIN4Traders! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- pip

### Setting Up Development Environment

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/BITCOIN4Traders.git
cd BITCOIN4Traders

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install in development mode
pip install -e .

# 5. Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Keep functions focused and small

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_antibias_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Code Quality Checks

```bash
# Format code with black
black src/ tests/

# Check with flake8
flake8 src/ tests/ --max-line-length=100

# Type checking with mypy
mypy src/
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvements

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 📁 Project Structure Guidelines

When adding new code, follow the existing structure:

```
src/
├── data/           # Data loading and management
├── features/       # Feature engineering
├── environment/    # Trading environment
├── math_tools/     # Mathematical models
├── risk/           # Risk management
├── agents/         # RL agents
├── training/       # Training infrastructure
├── backtesting/    # Backtesting & validation
├── validation/     # Anti-bias validation
├── costs/          # Transaction costs
├── reward/         # Reward functions
├── evaluation/     # Statistical evaluation
├── orders/         # Order management
├── execution/      # Live execution
├── connectors/     # Exchange connectors
└── monitoring/     # System monitoring
```

## 🧪 Testing Guidelines

### Writing Tests

All new code should have tests. Example:

```python
# tests/test_your_feature.py
import pytest
from src.your_module import YourClass

class TestYourFeature:
    def test_basic_functionality(self):
        obj = YourClass()
        result = obj.do_something()
        assert result == expected_value
    
    def test_edge_cases(self):
        # Test edge cases
        pass
    
    def test_error_handling(self):
        # Test error conditions
        with pytest.raises(ValueError):
            YourClass(invalid_param)
```

### Test Coverage

Aim for at least 80% code coverage for new features.

## 📝 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Short description of the function.
    
    Longer description if needed. Can span multiple lines
    and explain the function in detail.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> function_name(10, "test")
        True
    """
    pass
```

### README Updates

If you add new features, update the README.md with:
- Feature description
- Usage example
- Any new dependencies

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Step-by-step instructions
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, etc.
6. **Code Example**: Minimal code to reproduce the issue

Example:

```markdown
**Bug Description**
The backtesting engine crashes when processing empty dataframes.

**Steps to Reproduce**
1. Load empty dataframe
2. Run backtest
3. System crashes

**Expected Behavior**
Should return empty results or raise informative error.

**Actual Behavior**
System crashes with IndexError.

**Environment**
- Python: 3.11.4
- OS: Ubuntu 22.04
- BITCOIN4Traders: v1.0.0

**Code Example**
```python
import pandas as pd
from src.backtesting import WalkForwardEngine

df = pd.DataFrame()
engine = WalkForwardEngine()
engine.run(df)  # Crashes here
```
```

## 💡 Feature Requests

For feature requests, please:

1. Check if the feature already exists
2. Describe the use case
3. Explain why it would be useful
4. Provide example usage if possible

## 🎨 Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use descriptive variable names
- Add comments for complex logic

### Example

```python
from typing import Optional, List
import numpy as np

class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str, capital: float = 10000.0):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            capital: Initial capital
        """
        self.name = name
        self.capital = capital
        self.positions: List[float] = []
    
    def calculate_position_size(
        self, 
        signal: float, 
        volatility: Optional[float] = None
    ) -> float:
        """Calculate position size based on signal.
        
        Args:
            signal: Trading signal (-1 to 1)
            volatility: Optional volatility estimate
            
        Returns:
            Position size as fraction of capital
        """
        if volatility is None:
            volatility = 0.02  # Default 2%
        
        # Kelly criterion sizing
        kelly_fraction = signal / (volatility ** 2)
        
        # Limit to 25% max position
        return np.clip(kelly_fraction, -0.25, 0.25)
```

## 🔒 Security

- Never commit API keys or passwords
- Use environment variables for sensitive data
- Sanitize user inputs
- Follow security best practices

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in the README.md file.

## 📞 Questions?

If you have questions:
- Open an issue on GitHub
- Check existing documentation
- Review closed issues

Thank you for contributing to BITCOIN4Traders! 🚀
