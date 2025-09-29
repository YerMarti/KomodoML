import inspect

class BaseModel:
    """
    Generic ML model wrapper that provides a consistent interface
    for all models implemented.
    """

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        """Forward missing attributes/methods to the underlying model."""
        return getattr(self.model, name)

    def fit(self, X, y=None, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def score(self, X, y=None, **kwargs):
        # Not all models implement this, so fallback is optional
        if hasattr(self.model, "score"):
            return self.model.score(X, y, **kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} has no 'score' method.")
