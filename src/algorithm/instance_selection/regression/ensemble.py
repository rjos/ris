import numpy as np
from math import ceil
from collections import Counter


def ensemble_selection(data, targets, l, p, z, is_alg):
    n, m = data.shape
    sample_size = int(ceil(n * p))
    sample_size = n if sample_size > n else sample_size
    cnt = Counter()
    for i in range(l):
        # Bootstrap
        choice = np.random.choice(n, sample_size)

        # Apply
        selection = is_alg(data[choice, :], targets[choice])
        # TODO: If instance is selected twice, how many votes do we account?
        # Count the selected elements
        for element in np.unique(choice[selection]):
            cnt[element] += 1

    ret = []
    z = int(z * cnt.most_common(1)[0][1])
    for element, votes in cnt.most_common():
        if votes <= z:
            break
        ret.append(element)
    return ret
