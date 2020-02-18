from .utils.discretize import equal_binning_loue


def discretization(data, target, isalg, b):
    # Discretize
    discret = equal_binning_loue(target, b)
    # Run selection on the new target
    return isalg(data, discret)
