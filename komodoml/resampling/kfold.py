from .base import ResamplingStrategy
from sklearn.model_selection import KFold, cross_val_score

class KFoldFit(ResamplingStrategy):
    """
    K-Fold resampling strategy.

    Parameters
    ----------
    k : int, default=5
        Number of folds.
    scorer : str or callable, default=None
        Scoring function compatible with sklearn's `cross_val_score`.
    **kf_kwargs : dict
        Additional keyword arguments forwarded to sklearn's KFold,
        e.g., shuffle, random_state.
    """

    def __init__(self, k=5, scorer=None, **kf_kwargs):
        self.k = k
        self.scorer = scorer
        self.kf_kwargs = kf_kwargs
        self.scores_ = None

    def fit(self, model, X, y, **kwargs):
        splitter = KFold(n_splits=self.k, **self.kf_kwargs)
        self.scores_ = cross_val_score(model, X, y, cv=splitter, scoring=self.scorer)
        return model.fit(X, y, **kwargs)