import numpy as np

def random_missing(X, rate):
    Xc = X.copy()
    mask = np.random.rand(*X.shape) < rate
    Xc[mask] = np.nan
    return Xc

def column_missing(X, cols, rate):
    Xc = X.copy()
    drop_cols = np.random.choice(cols, int(len(cols)*rate), replace=False)
    Xc[drop_cols] = np.nan
    return Xc

def add_noise(X, num_cols, std):
    Xn = X.copy()
    for col in num_cols:
        Xn[col] += np.random.normal(0, std, len(X))
    return Xn

