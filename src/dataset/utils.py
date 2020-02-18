import numpy as np
from sklearn.preprocessing import MinMaxScaler


def to_numpy(data, dtype=np.float64):
    m = len(data)
    assert m, 'Should contain at least one variable'
    n = len(data[0])
    assert n, 'Should contain at least one instance'

    # TODO Check if its faster than transpose
    r = np.empty((n, m), dtype=dtype)
    for j, c in enumerate(data):
        for i, v in enumerate(c):
            r[i, j] = v
    return r

def to_data(a, b, axis=0):
    return np.transpose(np.concatenate((a,b), axis=axis)).tolist()

def normalize(data):
    MinMaxScaler(copy=False).fit_transform(data)
