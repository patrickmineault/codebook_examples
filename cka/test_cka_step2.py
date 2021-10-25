from cka_step2 import cka
import numpy as np
import pytest

def _get_one():
    X = np.cos(.1 * np.pi * np.arange(10)).reshape((-1, 1))
    Y = np.cos(2 + .07 * np.pi * np.arange(10)).reshape((-1, 1))
    return X, Y

def _get_multi():
    X = np.cos(.1 * np.pi * np.arange(10).reshape((-1, 1)) * np.linspace(.5, 1.5, num=3).reshape((1, -1)))
    Y = np.cos(.5 + .07 * np.pi * np.arange(10).reshape((-1, 1)) * np.linspace(.7, 1.3, num=4).reshape((1, -1)))
    return X, Y

def test_identity_lenient():
    """Create a random matrix and check it is perfectly correlated with itself."""
    X, _ = _get_multi()
    np.testing.assert_allclose(cka(X, X), 1.0)

def test_column_swaps():
    """Check that a matrix is perfectly correlated with itself even with column swaps."""
    X, _ = _get_multi()
    c = cka(X[:, [0, 1]], X[:, [1, 0]])
    np.testing.assert_allclose(c, 1.0)

def test_centering():
    """Check that a matrix is perfectly correlated with itself even with adding column offsets."""
    X, _ = _get_multi()
    Xp = X.copy()
    Xp[:, 1] += 1.0

    c = cka(X, Xp)
    np.testing.assert_allclose(c, 1.0)

def test_pure():
    """Check that a function doesn't change the original matrices."""
    X, _ = _get_multi()
    Xp = X.copy()
    Xp[:, 1] += 1.0

    Xp_original = Xp.copy()
    c = cka(X, Xp)
    np.testing.assert_allclose(Xp_original[:, 1], Xp[:, 1])

def test_corr():
    """The CKA of two vectors is the square of the correlation coefficient"""
    X, Y = _get_one()
    c1 = cka(X, Y)
    c2 = np.corrcoef(X.squeeze(), Y.squeeze())[0, 1] ** 2
    np.testing.assert_allclose(c1, c2)

def test_isoscaling():
    """CKA is insensitive to scaling by a scalar"""
    X, Y = _get_multi()
    c1 = cka(X, Y)
    c2 = cka(2.0 * X, - 1 * Y)
    np.testing.assert_allclose(c1, c2)

def test_rotation():
    """CKA is insensitive to rotations"""
    X, Y = _get_multi()
    X0 = X[:, :2]
    X0p = X0 @ np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    c1 = cka(X0, Y)
    c2 = cka(X0p, Y)
    np.testing.assert_allclose(c1, c2)

def test_no_iso():
    """CKA is sensitive to column scaling"""
    X, Y = _get_multi()
    X0 = X[:, :2]
    X0p = X0 @ np.array([[1, 1], [10, 1]])
    c1 = cka(X0, Y)
    c2 = cka(X0p, Y)
    assert abs(c1 - c2) > .001