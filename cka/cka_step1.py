import numpy as np

def cka(X, Y):
    # Implements linear CKA as in Kornblith et al. (2019)

    # Center X and Y
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # Calculate CKA
    XTX = X.T.dot(X)
    YTY = Y.T.dot(Y)
    YTX = Y.T.dot(X)

    return (YTX ** 2).sum() / np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
