import numpy as np
import logging
from .discretization import discretization
from ..classification.enn import enn


def tenn(training, targets, alpha, model, nn):
    n, m = training.shape

    t_s = set(range(n))
    for i in range(n):
        instance = training[i, :].reshape(1, m)
        target = targets[i]

        # KNN Model setup
        idx = np.array(list(t_s))
        nn.fit(training[idx, :])

        # Finding std
        neighbors = nn.kneighbors(instance, return_distance=False)
        # logging.debug(neighbors[0])
        std = np.std(targets[idx[neighbors[0]]])
        # Computing theta
        theta = alpha * std
        # Regression model setup
        t_s.remove(i)
        if t_s:
            idx = np.array(list(t_s))
            # print(idx.shape)
            train = training[idx, :]
            targ = targets[idx]
            model.fit(train, targ)

            # Regression model prediction
            prediction = model.predict(instance)[0]

            if abs(prediction - target) <= theta:
                t_s.add(i)
        else:
            t_s.add(i)

    return list(t_s)


def denn(training, target, d, nn_model):
    def _in_enn(data, labels):
        return enn(data, labels, nn_model)

    return discretization(training, target, _in_enn, d)
