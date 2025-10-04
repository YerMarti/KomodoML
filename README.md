[![PyPI version](https://img.shields.io/pypi/v/komodoml.svg)](https://pypi.org/project/komodoml/)
[![GitHub release](https://img.shields.io/github/v/release/YerMarti/KomodoML)](https://github.com/YerMarti/KomodoML/releases)
[![Autodocs](https://github.com/YerMarti/KomodoML/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/YerMarti/KomodoML/actions/workflows/docs.yml)

# KomodoML 🐊

A unified ML toolkit for models, metrics, plotting, and interpretability.

## Installation

### Install from PyPI

To install the latest version run:

```
pip install komodoml
```

### Install from GitHub

To install the latest version from the main branch:

```
pip install git+https://github.com/YerMarti/KomodoML.git
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

Alternatively, you can throw the resampling strategy into the model, whatever suits you best.

```python
kfold = KFoldFit(k=5)
clf = DecisionTreeClf(resampling=kfold)
clf.fit(X, y)
print(kfold.scores_)
```

## Documentation

* You can check the project's documentation at https://yermarti.github.io/KomodoML/

## Other resources

* PyPI: https://pypi.org/project/komodoml/