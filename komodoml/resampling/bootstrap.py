import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import get_scorer
from sklearn.base import clone
from .base import ResamplingStrategy


class BootstrapFit(ResamplingStrategy):
    """
    Bootstrap resampling strategy.

    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples.
    statistic : str, callable, or None, default=None
        Function to evaluate on resampled data. Should accept (model, X, y) and return a scalar.
        - str: scorer name from sklearn.metrics.get_scorer (e.g. "accuracy").
        - callable: function with signature (model, X, y) -> float.
        - None: use `model.score()` if available, else error.
    confidence_level : float, default=0.95
        Confidence level for the interval.
    random_state : int or None, default=None
        Random seed for reproducibility.
    method : {"basic", "percentile", "BCa"}, default="BCa"
        Method used to compute confidence intervals.
    """

    def __init__(self, n_resamples=1000, statistic=None, confidence_level=0.95,
                 random_state=None, method="BCa"):
        self.n_resamples = n_resamples
        self.statistic = statistic
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.method = method

        self.confidence_interval_ = None
        self.standard_error_ = None
        self.bias_ = None
        self.distribution_ = None

    def fit(self, model, X, y, **kwargs):
        if self.statistic is None:
            if not hasattr(model, "score"):
                raise ValueError(
                    f"{model.__class__.__name__} does not implement 'score'. "
                    "Please provide a custom statistic function or scorer string."
                )

            def stat(data, labels):
                m = clone(model)
                m.fit(data, labels, **kwargs)
                return m.score(data, labels)

        elif callable(self.statistic):
            def stat(data, labels):
                m = clone(model)
                m.fit(data, labels, **kwargs)
                return self.statistic(m, data, labels)

        elif isinstance(self.statistic, str):
            try:
                scorer = get_scorer(self.statistic)
            except KeyError:
                from sklearn.metrics import get_scorer_names
                raise ValueError(
                    f"Invalid scorer string '{self.statistic}'. "
                    f"Valid options are: {get_scorer_names()}"
                )

            def stat(data, labels):
                m = clone(model)
                m.fit(data, labels, **kwargs)
                return scorer(m, data, labels)

        else:
            raise TypeError("statistic must be None, a callable, or a string scorer name")
    
        X, y = np.asarray(X), np.asarray(y)

        res = bootstrap(
            data=(X, y),
            statistic=lambda data, labels: stat(data, labels),
            vectorized=False,
            paired=True,
            n_resamples=self.n_resamples,
            confidence_level=self.confidence_level,
            method=self.method,
            random_state=self.random_state
        )

        self.confidence_interval_ = res.confidence_interval
        self.standard_error_ = res.standard_error
        self.distribution_ = res.bootstrap_distribution
        self.bias_ = np.mean(res.bootstrap_distribution) - stat(X, y)

        return model.fit(X, y, **kwargs)