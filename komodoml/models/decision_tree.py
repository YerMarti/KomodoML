from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from komodoml.base import BaseModel, Wrapper

class DecisionTreeClf(BaseModel, metaclass=Wrapper):
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
    wrapped_cls = DecisionTreeClassifier


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = DecisionTreeClassifier(**kwargs)

class DecisionTreeReg(BaseModel, metaclass=Wrapper):
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
    wrapped_cls = DecisionTreeRegressor


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = DecisionTreeRegressor(**kwargs)