from cka_step1 import cka
import numpy as np
import pytest

def test_identity_lenient():
    # Create a random matrix and check it is perfectly correlated with itself.
    X = np.random.randn(100, 2)
    np.testing.assert_allclose(cka(X, X), 1.0)

def test_column_swaps():
    # Check that a matrix is perfectly correlated with itself even with column swaps.
    X = np.random.randn(100, 2)
    c = cka(X[:, [0, 1]], X[:, [1, 0]])
    np.testing.assert_allclose(c, 1.0)

def test_centering():
    # Check that a matrix is perfectly correlated with itself even with adding column offsets
    X = np.random.randn(100, 2)
    Xp = X.copy()
    Xp[:, 1] += 1.0

    c = cka(X, Xp)
    np.testing.assert_allclose(c, 1.0)

def test_pure():
    # Check that a function doesn't change the original matrices
    X = np.random.randn(100, 2)
    Xp = X.copy()
    Xp[:, 1] += 1.0

    Xp_original = Xp.copy()
    c = cka(X, Xp)
    np.testing.assert_allclose(Xp_original[:, 1], Xp[:, 1])