from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from komodoml.base import BaseModel


class DecisionTreeClf(BaseModel):
    """
    Decision Tree Classifier model.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments to pass to the DecisionTreeClassifier.

    Attributes
    ----------
    model : DecisionTreeClassifier
        The underlying Decision Tree Classifier model.
    """

    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)
        super().__init__(self.model)


class DecisionTreeReg(BaseModel):
    """
    Decision Tree Regressor model.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments to pass to the DecisionTreeRegressor.

    Attributes
    ----------
    model : DecisionTreeRegressor
        The underlying Decision Tree Regressor model.
    """

    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)
        super().__init__(self.model)