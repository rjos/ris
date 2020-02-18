from src.algorithm.instance_selection.classification.ris.ris import ris1, ris2, ris3
from .discretization import discretization


def get_ris(alg, thold):
    def _(data, labels):
        return alg(data, labels, thold)[0]

    return _


def dris1(training, target, d, threshold):
    return discretization(training, target, get_ris(ris1, threshold), d)


def dris2(training, target, d, threshold):
    return discretization(training, target, get_ris(ris2, threshold), d)


def dris3(training, target, d, threshold):
    return discretization(training, target, get_ris(ris3, threshold), d)
