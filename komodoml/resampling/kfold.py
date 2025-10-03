from .base import ResamplingStrategy, _cross_val_fit
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
    save_models : bool, default=False
        Whether to save the fitted models from each fold. If True, fitted models
        will be stored in the `models_` attribute.
    **kf_kwargs : dict
        Additional keyword arguments forwarded to sklearn's KFold,
        e.g., shuffle, random_state.
    """

    def __init__(self, k=5, scorer=None, save_models=False, **kf_kwargs):
        self.scorer = scorer
        self.scores_ = None
        self._splitter = KFold(n_splits=k, **kf_kwargs)
        self.save_models = save_models
        self.models_ = []

    def fit(self, model, X, y, **kwargs):
        if self.save_models:
            self.scores_, self.models_ = _cross_val_fit(model, X, y, cv=self._splitter, scoring=self.scorer, **kwargs)
        else:
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
    save_models : bool, default=False
        Whether to save the fitted models from each fold. If True, fitted models
        will be stored in the `models_` attribute.
    **skf_kwargs : dict
        Additional keyword arguments forwarded to sklearn's StratifiedKFold,
        e.g., shuffle, random_state.
    """

    def __init__(self, k=5, scorer=None, save_models=False, **skf_kwargs):
        self.scorer = scorer
        self.scores_ = None
        self._splitter = StratifiedKFold(n_splits=k, **skf_kwargs)
        self.save_models = save_models
        self.models_ = []

    def fit(self, model, X, y, **kwargs):
        if self.save_models:
            self.scores_, self.models_ = _cross_val_fit(model, X, y, cv=self._splitter, scoring=self.scorer, **kwargs)
        else:
            self.scores_ = cross_val_score(model, X, y, cv=self._splitter, scoring=self.scorer)
        return model.fit(X, y, **kwargs)


class LeaveOneOutFit(ResamplingStrategy):
    """
    Leave-One-Out resampling strategy.

    Parameters
    ----------
    scorer : str or callable, default=None
        Scoring function compatible with sklearn's `cross_val_score`.
    save_models : bool, default=False
        Whether to save the fitted models from each fold. If True, fitted models
        will be stored in the `models_` attribute.
    """

    def __init__(self, scorer=None, save_models=False):
        self.scorer = scorer
        self.scores_ = None
        self.save_models = save_models
        self._splitter = LeaveOneOut()

    def fit(self, model, X, y, **kwargs):
        if self.save_models:
            self.scores_, self.models_ = _cross_val_fit(model, X, y, cv=self._splitter, scoring=self.scorer, **kwargs)
        else:
            self.scores_ = cross_val_score(model, X, y, cv=self._splitter, scoring=self.scorer)
        return model.fit(X, y, **kwargs)