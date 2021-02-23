from collections import defaultdict
from math import exp
from math import inf
import timeit
import numpy as np

from src.algorithm.instance_selection.classification.ris.ris_helper import euclidean, _scores_radius as sr, \
    recompute_radius as rr, relevants as _relevants
    
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

mm = MinMaxScaler(copy=False)

__all__ = ['ris1', 'ris2', 'ris3']

def _scores_radius(dataset, targets, n_jobs=1):
    scores = np.zeros((targets.size, 1), dtype=np.float_)
    radius = np.zeros_like(scores)
    # Calculate score
    for i, (instance, target) in enumerate(zip(dataset, targets)):
        denom = 0
        num = 0
        radius[i] = np.inf
        for another_instance, another_target in zip(dataset, targets):
            dist = euclidean(instance, another_instance)
            val = exp(-dist)
            denom += val
            if target == another_target:
                num += val
            else:
                num -= val
                if dist < radius[i]:
                    radius[i] = dist
        num -= 1
        denom -= 1
        scores[i] = num / denom
    return scores, radius

def slow_relevants(idx, scores, radius, threshold, dataset, targets):
    selected_set = []
    selected_radius = []
    for id in idx:
        score = scores[id]
        if score < threshold:
            break
        instance = dataset[id, :]
        target = targets[id]

        for s_id in selected_set:

            another_target = targets[s_id]
            if another_target == target:
                dist = euclidean(instance, dataset[s_id, :])
                if dist < radius[s_id]:
                    break
        else:
            selected_set.append(id)
            selected_radius.append(radius[id])

    return selected_set, selected_radius

def _norm_by_clas(scores, targets):
    # Normalize by class
    classes = np.unique(targets)
    for class_ in classes:
        mm_class_ = MinMaxScaler()

        mask = (targets == class_)
        scores[mask] = mm_class_.fit_transform(scores[mask])

def ris1(dataset, targets, threshold, n_jobs=1):
    scores, radius = sr(dataset, targets, n_jobs)

    mm.fit_transform(scores)
    
    # Fixed decimals issues
    scores = np.round(scores, decimals=5)
    
    # Sort it
    idx = scores.argsort(None)[::-1]
    idx = idx.astype(int)

    # Return the relevants
    # _rels = slow_relevants(idx, scores, radius, threshold, dataset, targets)
    _rels = _relevants(idx, scores, radius, threshold, dataset, targets)

    return _rels

def ris2(dataset, targets, threshold, n_jobs=1):
    scores, radius = sr(dataset, targets, n_jobs)
    
    _norm_by_clas(scores, targets)
    
    # Fixed decimals issues
    scores = np.round(scores, decimals=5)

    idx = scores.argsort(None)[::-1]
    idx = idx.astype(int)

    # Return the relevants
    # _rels = slow_relevants(idx, scores, radius, threshold, dataset, targets)
    _rels = _relevants(idx, scores, radius, threshold, dataset, targets)

    return _rels

def ris3(dataset, targets, threshold, n_jobs=1):
    scores, radius = sr(dataset, targets, n_jobs)
    
    _norm_by_clas(scores, targets)

    # Fixed decimals issues
    scores = np.round(scores, 5)
    
    # Remove from idx the ones that have bad scores and recompute radius
    idx = scores.argsort(None)[::-1]
    idx = idx.astype(int)

    for i, v in enumerate(scores[idx]):
        if v < threshold:
            idx = idx[:i]
            break

    # Now recompute radius
    '''
    for id in idx:
        instance = dataset[id, :]
        target = targets[id]
        max_dist = inf
        for o_id in idx:
            o_instance = dataset[o_id, :]
            o_target = targets[o_id]
            if o_target != target:
                dist = euclidean(instance, o_instance)
                if dist < max_dist:
                    max_dist = dist
        radius[id] = max_dist
    '''
    rr(idx, radius, dataset, targets, n_jobs)

    # Return the relevants
    # _rels = slow_relevants(idx, scores, radius, threshold, dataset, targets)
    _rels = _relevants(idx, scores, radius, threshold, dataset, targets)

    return _rels

def slow_ris3(dataset, targets, threshold, n_jobs=1):
    scores, radius = _scores_radius(dataset, targets, n_jobs)
    _norm_by_clas(scores, targets)
    # Remove from idx the ones that have bad scores and recompute radius

    idx = scores.argsort(None)[::-1]
    for i, v in enumerate(scores[idx]):
        if v < threshold:
            idx = idx[:i]
            break

    # Now recompute radius

    for id in idx:
        instance = dataset[id, :]
        target = targets[id]
        max_dist = inf
        for o_id in idx:
            o_instance = dataset[o_id, :]
            o_target = targets[o_id]
            if o_target != target:
                dist = euclidean(instance, o_instance)
                if dist < max_dist:
                    max_dist = dist
        radius[id] = max_dist

    return slow_relevants(idx, scores, radius, threshold, dataset, targets)

def classify(X_test, X_selection, radius):

    # Sum const to avoid division by zero
    radius_temp = radius + 1e-8

    # Compute distance between Test and Selection instances
    dists = euclidean_distances(X_test, X_selection)
    dists = dists / radius_temp

    # Get index of train samples to get class label
    index = np.argmin(dists, axis=1)
    
    return index

if __name__ == '__main__':

    from sklearn.datasets import make_blobs

    x, y = make_blobs(1000, 2, centers=5)

    # slow_ris3(x, y, 0.1)
    s1, _ = ris2(x, y, 0.1)
    s2, _ = ris3(x, y, 0.1)
    print(len(s1), x.shape[0])
    print(len(s2), x.shape[0])
    a, _ = ris3(x, y, 0.1)
    b, _ = ris3(x, y, 0.1, 4)
    s = timeit.timeit(lambda: ris3(x, y, 0.1), number=1)
    s2 = timeit.timeit(lambda: ris3(x, y, 0.1, 2), number=1)
    s3 = timeit.timeit(lambda: ris3(x, y, 0.1, 3), number=1)
    s4 = timeit.timeit(lambda: ris3(x, y, 0.1, 4), number=1)
    print(s, s4)
    print(a, b, sep='\n')
