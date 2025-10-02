from komodoml.resampling import ResamplingStrategy

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

    def fit(self, X, y=None, resampling: ResamplingStrategy=None, **kwargs):
        """
        Fit the model to the data, optionally using a resampling strategy.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values.
        resampling : ResamplingStrategy, default=None
            Resampling strategy to use.
        **kwargs : additional keyword arguments
            Additional arguments passed to the model's fit method.
        """
        if resampling is not None:
            return resampling.fit(self.model, X, y, **kwargs)
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.
        **kwargs : additional keyword arguments
            Additional arguments passed to the model's predict method.
        """
        return self.model.predict(X, **kwargs)

    def score(self, X, y=None, **kwargs):
        # Not all models implement this, so fallback is optional
        if hasattr(self.model, "score"):
            return self.model.score(X, y, **kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} has no 'score' method.")