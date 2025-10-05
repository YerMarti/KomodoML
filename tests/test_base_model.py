import pytest
from unittest.mock import Mock
import numpy as np

from komodoml.base.base_model import BaseModel
from komodoml.resampling import ResamplingStrategy


class TestBaseModel:
    """Test suite for BaseModel class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock model for testing
        self.mock_model = Mock()
        self.mock_model.fit.return_value = self.mock_model
        self.mock_model.predict.return_value = np.array([1, 0, 1])
        self.mock_model.score.return_value = 0.85
        
        # Create BaseModel instance
        self.base_model = BaseModel(self.mock_model)

    def test_init(self):
        """Test BaseModel initialization."""
        model = Mock()
        base_model = BaseModel(model)
        assert base_model.model is model

    def test_getattr_forwards_to_model(self):
        """Test that missing attributes are forwarded to the underlying model."""
        # Set up a custom attribute on the mock model
        self.mock_model.custom_attribute = "test_value"
        self.mock_model.custom_method = Mock(return_value="method_result")
        
        # Test attribute forwarding
        assert self.base_model.custom_attribute == "test_value"
        
        # Test method forwarding
        result = self.base_model.custom_method("arg1", key="value")
        assert result == "method_result"
        self.mock_model.custom_method.assert_called_once_with("arg1", key="value")

    def test_fit_without_resampling(self, sample_data):
        """Test fit method without resampling strategy."""
        X, y = sample_data
        result = self.base_model.fit(X, y, param1="value1")
        
        # Check that the underlying model's fit was called correctly
        self.mock_model.fit.assert_called_once_with(X, y, param1="value1")
        assert result is self.mock_model

    def test_fit_with_resampling(self, sample_data):
        """Test fit method with resampling strategy."""
        # Create mock resampling strategy
        mock_resampling = Mock(spec=ResamplingStrategy)
        mock_resampling.fit.return_value = "resampling_result"
        
        X, y = sample_data
        result = self.base_model.fit(X, y, resampling=mock_resampling, param1="value1")
        
        # Check that resampling.fit was called instead of model.fit
        mock_resampling.fit.assert_called_once_with(self.mock_model, X, y, param1="value1")
        self.mock_model.fit.assert_not_called()
        assert result == "resampling_result"

    def test_predict(self, sample_data):
        """Test predict method."""
        X, _ = sample_data
        result = self.base_model.predict(X, param1="value1")
        
        self.mock_model.predict.assert_called_once_with(X, param1="value1")
        np.testing.assert_array_equal(result, np.array([1, 0, 1]))

    def test_score_with_score_method(self, sample_data):
        """Test score method when underlying model has score method."""
        X, y = sample_data
        result = self.base_model.score(X, y, param1="value1")
        
        self.mock_model.score.assert_called_once_with(X, y, param1="value1")
        assert result == 0.85

    def test_score_without_score_method(self, sample_data):
        """Test score method when underlying model doesn't have score method."""
        # Create a model without score method
        model_without_score = Mock()
        del model_without_score.score  # Remove score attribute
        base_model = BaseModel(model_without_score)
        
        X, y = sample_data
        with pytest.raises(NotImplementedError, match="BaseModel has no 'score' method"):
            base_model.score(X, y)

    def test_score_with_y_none(self, sample_data):
        """Test score method with y=None."""
        X, _ = sample_data
        result = self.base_model.score(X)
        
        self.mock_model.score.assert_called_once_with(X, None)
        assert result == 0.85

    def test_integration_with_real_sklearn_model(self):
        """Integration test with a real sklearn model."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import make_classification
            
            # Create sample data
            X, y = make_classification(n_samples=100, n_features=4, random_state=42)
            
            # Create and wrap sklearn model
            sklearn_model = LogisticRegression(random_state=42)
            base_model = BaseModel(sklearn_model)
            
            # Test fitting and prediction
            base_model.fit(X, y)
            predictions = base_model.predict(X)
            score = base_model.score(X, y)
            
            assert len(predictions) == len(y)
            assert isinstance(score, float)
            assert 0 <= score <= 1
            
            # Test that we can access sklearn-specific attributes
            assert hasattr(base_model, 'coef_')
            
        except ImportError:
            pytest.skip("scikit-learn not available for integration test")


class TestBaseModelEdgeCases:
    """Test edge cases and error conditions for BaseModel."""

    def test_init_with_none_model(self):
        """Test initialization with None model."""
        base_model = BaseModel(None)
        assert base_model.model is None

    def test_getattr_with_none_model(self):
        """Test attribute access when model is None."""
        base_model = BaseModel(None)
        
        with pytest.raises(AttributeError):
            _ = base_model.some_attribute

    def test_fit_with_none_y(self, sample_data):
        """Test fit method with y=None (unsupervised learning)."""
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        base_model = BaseModel(mock_model)
        
        X, _ = sample_data  # Use X from sample_data, ignore y
        result = base_model.fit(X)
        
        mock_model.fit.assert_called_once_with(X, None)

    def test_resampling_strategy_interface(self, sample_data):
        """Test that resampling parameter accepts ResamplingStrategy interface."""
        mock_model = Mock()
        base_model = BaseModel(mock_model)
        
        # Test with object that has fit method (duck typing)
        mock_resampling = Mock()
        mock_resampling.fit.return_value = "result"
        
        # Use sample data from conftest.py
        X, y = sample_data
        
        result = base_model.fit(X, y, resampling=mock_resampling)
        
        mock_resampling.fit.assert_called_once()
        assert result == "result"