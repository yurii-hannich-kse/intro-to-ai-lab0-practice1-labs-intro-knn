import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from grader_utils import load_notebook_functions

# Path to the notebook we grade
hw = load_notebook_functions("lab.ipynb")


def test_euclidean_distance():
    """Task 1: Euclidean distance (2 points)."""
    assert hasattr(hw, 'euclidean_distance'), "Function euclidean_distance not found!"

    d = hw.euclidean_distance(np.array([0, 0]), np.array([3, 4]))
    assert np.isclose(d, 5.0, atol=1e-6), f"Expected 5.0, got {d}"

    d2 = hw.euclidean_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert np.isclose(d2, np.sqrt(27), atol=1e-6), f"Expected sqrt(27), got {d2}"


def test_classify_example():
    """Task 2: Classify one example with KNN logic (2 points)."""
    assert hasattr(hw, 'classify_example'), "Function classify_example not found!"

    # Simple 1D toy dataset where nearest neighbor is obvious
    X_train = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_train = np.array([0, 0, 1, 1])

    pred_k1 = hw.classify_example(X_train, y_train, np.array([0.2]), k=1)
    assert pred_k1 == 0, f"Expected class 0 for k=1, got {pred_k1}"

    pred_k3 = hw.classify_example(X_train, y_train, np.array([10.4]), k=3)
    assert pred_k3 == 1, f"Expected class 1 for k=3, got {pred_k3}"


def test_predict():
    """Task 3: Predict for a batch (2 points)."""
    assert hasattr(hw, 'predict'), "Function predict not found!"

    # Toy dataset batch check
    X_train = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.2], [10.4], [0.9]])

    y_pred = hw.predict(X_train, y_train, X_test, k=1)
    assert isinstance(y_pred, (list, np.ndarray)), "predict must return list or numpy array"
    y_pred = np.array(y_pred)
    assert y_pred.shape == (3,), f"Expected shape (3,), got {y_pred.shape}"
    assert np.all(np.isin(y_pred, [0, 1])), f"Predictions must be 0/1 for toy data, got {y_pred}"

    # Iris sanity check: should be reasonably accurate
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    y_pred_iris = np.array(hw.predict(X_tr, y_tr, X_te, k=3))
    acc = accuracy_score(y_te, y_pred_iris)
    assert acc >= 0.90, f"Expected accuracy >= 0.90 on Iris, got {acc:.3f}"