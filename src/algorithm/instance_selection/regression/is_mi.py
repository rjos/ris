import numpy as np
from scipy.special import psi
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


# TODO Cythonize these
def x_metric(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
    #return euclidean_distances(a[:-1].reshape(1, -1), b[:-1].reshape(1, -1))[0][0]


def y_metric(a, b):
    return np.abs(a[-1] - b[-1])


def mi_metric(a, b):
    return max(x_metric(a, b), y_metric(a, b))


def mi(a, b, k):
    z = np.hstack((a, b))
    n, m = z.shape
    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='mi_distance')
    nn.fit(z)
    nn_x = NearestNeighbors(n_neighbors=k)
    nn_x.fit(a)
    nn_y = NearestNeighbors(n_neighbors=k)
    nn_y.fit(b)
    acc = 0

    neighbors = nn.kneighbors(return_distance=False)
    for current, neighbor in zip(z, neighbors):
        k_nearest = z[neighbor[-1], :]
        e_x = x_metric(current, k_nearest)
        e_y = y_metric(current, k_nearest)
        # print(current, k_nearest, e_x, e_y)

        # Fix because the same point is returned with a distance zero
        n_x = len(nn_x.radius_neighbors(current[:-1].reshape(1, -1), e_x, return_distance=False)[0]) - 1
        n_y = len(nn_y.radius_neighbors(current[-1], e_y, return_distance=False)[0]) - 1
        if n_x:
            acc += psi(n_x)
        if n_y:
            acc += psi(n_y)

    return psi(k) + psi(n) - 1 / k - (1 / n) * acc


def is_mi(training, target, k, alpha):
    n = len(target)
    # Find the k nearest neighbours to each instance of training
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(training)

    # Find the MI when removing i from instance
    I = np.empty((n,), dtype=np.float)
    for i in range(n):
        training_without_i = np.vstack((training[:i, :], training[i + 1:, :]))
        target_without_i = np.vstack((target[:i], target[i + 1:]))
        I[i] = mi(training_without_i, target_without_i, k)
    # Normalize
    if I.max() != I.min():
        I = (I - I.min()) / float(I.max() - I.min())

    ret = []

    neighbours = nn.kneighbors(return_distance=False)

    for i in range(n):
        z = I[i] - I[neighbours[i]]
        if not np.all(z > alpha):
            ret.append(i)
    return ret


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([2, 0, 0, 1])
    y.shape = 4, 1
    I = is_mi(x, y, 2, .5)
