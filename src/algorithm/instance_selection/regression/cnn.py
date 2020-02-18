import numpy as np
from .discretization import discretization
from ..classification.cnn import cnn


def tcnn(training, targets, alpha, model, nn):
    n, m = training.shape
    p = [0]
    t = set(range(n))
    for i in range(1, n):
        instance = training[i, :].reshape(1, m)
        target = targets[i]
        # KNN Model setup
        idx = np.array(list(t))
        nn.fit(training[idx, :])

        # Finding std
        neighbors = nn.kneighbors(instance, return_distance=False)
        std = np.std(targets[idx[neighbors[0]]])
        theta = alpha * std

        # Predict with P
        model.fit(training[p, :], targets[p])
        prediction = model.predict(instance)[0]

        if abs(prediction - target) > theta:
            t.remove(i)
            p.append(i)

    return p


def dcnn(training, target, d, nn_model):
    def _in_cnn(data, labels):
        return cnn(data, labels, nn_model)

    return discretization(training, target, _in_cnn, d)
