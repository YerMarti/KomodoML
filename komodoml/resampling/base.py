from komodoml.base.base_model import BaseModel


class ResamplingStrategy:
    """
    Base class for resampling-based training strategies.
    All resampling strategies should inherit from this class and implement the `fit` method.
    """

    def fit(self, model: BaseModel, X, y, **kwargs):
        """
        Fit the model using the resampling strategy.

        Parameters
        ----------
        model : BaseModel
            The model to fit.
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values.
        **kwargs : additional keyword arguments
            Additional arguments passed to the model's fit method.
        """
        raise NotImplementedError