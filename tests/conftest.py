"""
Shared pytest configuration and fixtures.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add the project root to Python path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_data():
    """Create sample data for testing machine learning models."""
    try:
        import numpy as np
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        return X, y
    except ImportError:
        # Fallback for when numpy is not available
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 1, 0, 1]
        return X, y


@pytest.fixture
def mock_model():
    """Create a mock model for testing across multiple test files."""
    try:
        import numpy as np
        model = Mock()
        model.fit.return_value = model
        model.predict.return_value = np.array([1, 0, 1])
        model.score.return_value = 0.85
        return model
    except ImportError:
        # Fallback for when numpy is not available
        model = Mock()
        model.fit.return_value = model
        model.predict.return_value = [1, 0, 1]
        model.score.return_value = 0.85
        return model