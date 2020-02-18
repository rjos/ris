from math import inf, log2
import numpy as np


def loui(histogram, width):
    entropy = 0
    for n in histogram:
        if n < 2:
            return inf
        else:
            entropy -= n * log2((n - 1) / width)
    return entropy


def equal_binning_loue(targets, b):
    if type(b) == int:
        b = range(1, b + 1)
    best_entropy = inf

    max_v = np.max(targets)
    min_v = np.min(targets)
    num_targets = len(targets)
    for num_bins in b:
        # Discretize
        width = (max_v - min_v) / num_bins

        histogram = np.zeros((num_bins,), dtype=np.int64)
        inv_target = np.empty((num_targets,), dtype=np.int64)
        for t, v in enumerate(targets):
            for j in range(num_bins):
                max_v_in_bin = min_v + (j + 1) * width
                if v <= max_v_in_bin:
                    histogram[j] += 1
                    inv_target[t] = j
                    break

        # Calculate the entropy
        entropy = loui(histogram, width)
        if entropy < best_entropy:
            #print('Num of bins', num_bins)
            best_entropy = entropy
            discretization = inv_target

    return discretization
