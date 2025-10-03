[![Autodocs](https://github.com/YerMarti/KomodoML/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/YerMarti/KomodoML/actions/workflows/docs.yml)

# KomodoML

A unified ML toolkit for models, metrics, plotting, and interpretability.

## Installation

Since KomodoML is not yet published on PyPI, you can install it directly from GitHub or set it up locally for development.

### Install from GitHub

To install the latest version from the main branch:

```
pip install git+https://github.com/YerMarti/KomodoML.git
```

### Local Development Installation

If you want to work on the library or run it in development mode:

```
git clone https://github.com/YerMarti/KomodoML.git
cd KomodoML
pip install -e .
```

## Quick start

The following snippet trains a Decision Tree Classifier model using a *k*-fold resampling strategy.

```python
from komodoml.models import DecisionTreeClf
from komodoml.resampling import KFoldFit

clf = DecisionTreeClf()
kfold = KFoldFit(k=5)
kfold.fit(clf, X, y)
print(kfold.scores_)
```

Alternatively, you can throw the resampling strategy into the model, whatever best suits you.

```python
kfold = KFoldFit(k=5)
clf = DecisionTreeClf(resampling=kfold)
clf.fit(X, y)
print(kfold.scores_)
```

## Documentation

* Documentation: https://yermarti.github.io/KomodoML/