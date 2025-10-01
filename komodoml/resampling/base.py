class ResamplingStrategy:
    """
    Base class for resampling-based training strategies.
    All resampling strategies should inherit from this class and implement the `fit` method.
    """

    def fit(self, model, X, y, **kwargs):
        raise NotImplementedError