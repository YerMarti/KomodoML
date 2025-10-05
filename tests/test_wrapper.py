import pytest
from unittest.mock import Mock
from komodoml.base.wrapper import Wrapper, _forward_methods


class TestForwardMethods:
    """Test suite for _forward_methods function."""

    def test_forward_all_methods(self):
        """Test forwarding all methods from source class."""
        
        class SourceClass:
            def method1(self, arg1, arg2="default"):
                return f"method1: {arg1}, {arg2}"
            
            def method2(self):
                return "method2"
        
        class TargetClass:
            def __init__(self):
                self.model = Mock()
                self.model.method1.return_value = "mocked_method1"
                self.model.method2.return_value = "mocked_method2"
        
        # Forward methods
        _forward_methods(SourceClass, TargetClass)
        
        # Test that methods were added
        assert hasattr(TargetClass, 'method1')
        assert hasattr(TargetClass, 'method2')
        
        # Test method calls
        target = TargetClass()
        result1 = target.method1("arg1", arg2="custom")
        result2 = target.method2()
        
        target.model.method1.assert_called_once_with("arg1", arg2="custom")
        target.model.method2.assert_called_once_with()
        assert result1 == "mocked_method1"
        assert result2 == "mocked_method2"

    def test_forward_specific_methods(self):
        """Test forwarding only specific methods."""
        
        class SourceClass:
            def method1(self):
                return "method1"
            
            def method2(self):
                return "method2"
            
            def method3(self):
                return "method3"
        
        class TargetClass:
            def __init__(self):
                self.model = Mock()
        
        # Forward only method1 and method2
        _forward_methods(SourceClass, TargetClass, methods=['method1', 'method2'])
        
        assert hasattr(TargetClass, 'method1')
        assert hasattr(TargetClass, 'method2')
        assert not hasattr(TargetClass, 'method3')

    def test_dont_overwrite_existing_methods(self):
        """Test that existing methods are not overwritten."""
        
        class SourceClass:
            def existing_method(self):
                return "from_source"
        
        class TargetClass:
            def existing_method(self):
                return "from_target"
        
        _forward_methods(SourceClass, TargetClass)
        
        target = TargetClass()
        result = target.existing_method()
        assert result == "from_target"

    def test_no_model_attribute(self):
        """Test behavior when target has no model attribute."""
        
        class SourceClass:
            def method1(self):
                return "method1"
        
        class TargetClass:
            pass
        
        _forward_methods(SourceClass, TargetClass)
        
        target = TargetClass()
        result = target.method1()
        assert result is None


class TestWrapperMetaclass:
    """Test suite for Wrapper metaclass."""

    def test_wrapper_with_wrapped_cls(self):
        """Test Wrapper metaclass with wrapped_cls specified."""
        
        class ModelToWrap:
            """Original model documentation."""
            
            def __init__(self, param1, param2="default"):
                self.param1 = param1
                self.param2 = param2
            
            def fit(self, X, y):
                return "fitted"
            
            def predict(self, X):
                return "predictions"
            
            def custom_method(self, arg):
                return f"custom: {arg}"
        
        class WrappedModel(metaclass=Wrapper):
            """Wrapper documentation."""
            wrapped_cls = ModelToWrap
            
            def __init__(self, *args, **kwargs):
                self.model = ModelToWrap(*args, **kwargs)
        
        # Test that methods were forwarded
        assert hasattr(WrappedModel, 'fit')
        assert hasattr(WrappedModel, 'predict')
        assert hasattr(WrappedModel, 'custom_method')
        
        # Test method calls
        wrapped = WrappedModel("param1_value", param2="custom")
        
        # Mock the underlying model to test forwarding
        wrapped.model = Mock()
        wrapped.model.fit.return_value = "mocked_fit"
        wrapped.model.predict.return_value = "mocked_predict"
        wrapped.model.custom_method.return_value = "mocked_custom"
        
        assert wrapped.fit("X", "y") == "mocked_fit"
        assert wrapped.predict("X") == "mocked_predict"
        assert wrapped.custom_method("arg") == "mocked_custom"
        
        wrapped.model.fit.assert_called_once_with("X", "y")
        wrapped.model.predict.assert_called_once_with("X")
        wrapped.model.custom_method.assert_called_once_with("arg")

    def test_wrapper_without_wrapped_cls(self):
        """Test Wrapper metaclass when wrapped_cls is not specified."""
        
        class SimpleClass(metaclass=Wrapper):
            def existing_method(self):
                return "existing"
        
        # Should work normally without forwarding
        simple = SimpleClass()
        assert simple.existing_method() == "existing"

    def test_docstring_merging(self):
        """Test that docstrings are merged correctly."""
        
        class ModelToWrap:
            """Original model documentation."""
            pass
        
        class WrappedModel(metaclass=Wrapper):
            """Wrapper documentation."""
            wrapped_cls = ModelToWrap
        
        expected_doc = "Wrapper documentation.\n\nWrapped class docstring:\nOriginal model documentation."
        assert WrappedModel.__doc__ == expected_doc

    def test_docstring_merging_with_none_docs(self):
        """Test docstring merging when one or both docstrings are None."""
        
        class ModelToWrap:
            pass  # No docstring
        
        class WrappedModel(metaclass=Wrapper):
            """Wrapper documentation."""
            wrapped_cls = ModelToWrap
        
        expected_doc = "Wrapper documentation.\n\nWrapped class docstring:\n"
        assert WrappedModel.__doc__ == expected_doc

    def test_signature_attachment(self):
        """Test that __init__ signature is attached from wrapped class."""
        
        class ModelToWrap:
            def __init__(self, param1: int, param2: str = "default", *args, **kwargs):
                pass
        
        class WrappedModel(metaclass=Wrapper):
            wrapped_cls = ModelToWrap
        
        # Test that signature was attached
        assert hasattr(WrappedModel, '__signature__')
        sig = WrappedModel.__signature__
        
        # Check parameters (excluding 'self')
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert 'param1' in params
        assert 'param2' in params

    def test_signature_attachment_failure(self):
        """Test graceful handling when signature attachment fails."""
        
        # Create a class with problematic __init__ for signature inspection
        class ProblematicModel:
            pass
        
        # Remove __init__ to cause signature inspection to fail
        delattr(ProblematicModel, '__init__')
        
        class WrappedModel(metaclass=Wrapper):
            wrapped_cls = ProblematicModel
        
        # Should not crash, just skip signature attachment
        assert WrappedModel is not None


class TestWrapperIntegration:
    """Integration tests for the Wrapper metaclass."""

    def test_real_world_usage(self):
        """Test realistic usage of Wrapper metaclass."""
        
        class MockSklearnModel:
            """Mock scikit-learn like model."""
            
            def __init__(self, learning_rate=0.01, max_iter=100):
                self.learning_rate = learning_rate
                self.max_iter = max_iter
                self.is_fitted = False
            
            def fit(self, X, y, sample_weight=None):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                return ["prediction"] * len(X)
            
            def score(self, X, y):
                return 0.95
            
            def get_params(self, deep=True):
                return {"learning_rate": self.learning_rate, "max_iter": self.max_iter}
        
        class EnhancedModel(metaclass=Wrapper):
            """Enhanced model with additional functionality."""
            wrapped_cls = MockSklearnModel
            
            def __init__(self, *args, **kwargs):
                self.model = MockSklearnModel(*args, **kwargs)
                self.training_history = []
            
            def fit(self, X, y, **kwargs):
                # Override fit to add logging
                result = self.model.fit(X, y, **kwargs)
                self.training_history.append({"X_shape": X.shape if hasattr(X, 'shape') else len(X)})
                return result
            
            def get_training_history(self):
                return self.training_history
        
        # Test the enhanced model
        model = EnhancedModel(learning_rate=0.05, max_iter=200)
        
        # Test that wrapped methods work
        assert model.get_params() == {"learning_rate": 0.05, "max_iter": 200}
        
        # Test overridden method
        X, y = [[1, 2], [3, 4]], [0, 1]
        model.fit(X, y)
        
        assert len(model.training_history) == 1
        assert model.training_history[0]["X_shape"] == 2
        
        # Test forwarded methods
        predictions = model.predict(X)
        assert predictions == ["prediction", "prediction"]
        
        score = model.score(X, y)
        assert score == 0.95