from komodoml.base.base_model import BaseModel
from sklearn.base import clone
import numpy as np


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


def _cross_val_fit(model, X, y, splitter, scorer=None, **fit_kwargs):
    """
    Custom cross-validation loop that can store trained models.
    
    Parameters
    ----------
    model : BaseModel
        The model to fit.
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    splitter : cross-validation splitter
        An object that provides train/test indices to split data into training and test sets.
    scorer : callable, default=None
        A function to evaluate the predictions on the test set. If None, the model's `score` method is used.
    **fit_kwargs : additional keyword arguments
        Additional arguments passed to the model's fit method.
    
    Returns
    -------
    scores : array, shape (n_splits,)
        Array of scores for each fold.
    models : list of BaseModel
        List of fitted models for each fold.
    """
    scores = []
    models = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m = clone(model)
        m.fit(X_train, y_train, **fit_kwargs)

        if scorer:
            score = scorer(m, X_test, y_test)
        elif hasattr(m, "score"):
            score = m.score(X_test, y_test)
        else:
            raise ValueError("Model has no 'score' method and no scorer was provided.")

        scores.append(score)
        models.append(m)

    return np.array(scores), models