from .base import ResamplingStrategy
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score

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
        self.scorer = scorer
        self.scores_ = None
        self._splitter = KFold(n_splits=k, **kf_kwargs)

    def fit(self, model, X, y, **kwargs):
        self.scores_ = cross_val_score(model, X, y, cv=self._splitter, scoring=self.scorer)
        return model.fit(X, y, **kwargs)


class StratifiedKFoldFit(ResamplingStrategy):
    """
    Stratified K-Fold resampling strategy.

    Parameters
    ----------
    k : int, default=5
        Number of folds.
    scorer : str or callable, default=None
        Scoring function compatible with sklearn's `cross_val_score`.
    **skf_kwargs : dict
        Additional keyword arguments forwarded to sklearn's StratifiedKFold,
        e.g., shuffle, random_state.
    """

    def __init__(self, k=5, scorer=None, **skf_kwargs):
        self.scorer = scorer
        self.scores_ = None
        self._splitter = StratifiedKFold(n_splits=k, **skf_kwargs)

    def fit(self, model, X, y, **kwargs):
        self.scores_ = cross_val_score(model, X, y, cv=self._splitter, scoring=self.scorer)
        return model.fit(X, y, **kwargs)


class LeaveOneOutFit(ResamplingStrategy):
    """
    Leave-One-Out resampling strategy.

    Parameters
    ----------
    scorer : str or callable, default=None
        Scoring function compatible with sklearn's `cross_val_score`.
    """

    def __init__(self, scorer=None):
        self.scorer = scorer
        self.scores_ = None
        self._splitter = LeaveOneOut()

    def fit(self, model, X, y, **kwargs):
        self.scores_ = cross_val_score(model, X, y, cv=self._splitter, scoring=self.scorer)
        return model.fit(X, y, **kwargs)