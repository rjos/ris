#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport numpy as np

ctypedef np.double_t DTYPE_t
ctypedef np.int_t INT_t

from libc.math cimport sqrt, exp
from numpy.math cimport INFINITY

from cython.parallel import prange


cdef DTYPE_t c_euclidean(DTYPE_t* a, DTYPE_t* b, INT_t s) nogil:
    cdef:
        INT_t c = 0
        DTYPE_t v = 0
    for c in range(s):
        v += (a[c] - b[c]) ** 2
    return sqrt(v)

def euclidean(DTYPE_t[:] a, DTYPE_t[:] b):
    return c_euclidean(&a[0], &b[0], a.shape[0])

def classify(DTYPE_t[:, :] test not None, DTYPE_t[:, :] selection not None, DTYPE_t[:] radius):
    cdef:
        np.ndarray[INT_t, ndim=1] predictions = np.zeros((test.shape[0],), dtype=np.int_)
        DTYPE_t* x_test
        DTYPE_t* x_sel
        INT_t n, m, s, idx
        DTYPE_t d_min
    
    n, m = test.shape[0], test.shape[1]
    s = selection.shape[0]

    for i in range(n):
        x_test = &test[i, 0]
        idx = 0
        d_min = INFINITY
        for j in range(s):
            x_sel = &selection[j, 0]
            r = radius[j]

            d = c_euclidean(x_test, x_sel, m)
            d = (d / r)

            if d < d_min:
                idx = j
                d_min = d

        predictions[i] = idx
    return predictions

def relevants(INT_t[:] idx not None, DTYPE_t[:, :] scores not None, DTYPE_t[:, :] radius not None, DTYPE_t threshold, DTYPE_t[:, :] dataset not None, INT_t[:] targets not None):
    cdef:
        list selected_set = []
        list selected_radius = []
        DTYPE_t score, dist, r
        INT_t n, m, s, id, s_id
        DTYPE_t* instance
        DTYPE_t* another_instance
        INT_t target, another_target
    
    n, m = dataset.shape[0], dataset.shape[1]
    s = len(selected_set)

    for i in range(n):
        id = idx[i]
        score = scores[id, 0]

        if score < threshold:
            break
        
        instance = &dataset[id, 0]
        target = targets[id]

        for j in range(s):

            s_id = selected_set[j]
            
            another_instance = &dataset[s_id, 0]
            another_target = targets[s_id]
            r = radius[s_id, 0]
            
            if another_target == target:
                dist = c_euclidean(instance, another_instance, m)
                
                if dist < r:
                    break
        else:
            selected_set.append(id)
            selected_radius.append(radius[id])
            s = len(selected_set)
    
    return selected_set, selected_radius

def recompute_radius(INT_t[:] idx not None, DTYPE_t[:, :] radius not None, DTYPE_t[:, :] dataset not None, INT_t[:] targets not None, INT_t n_jobs=1):
    cdef:
        INT_t i, j, id, o_id, n, m
        DTYPE_t* instance
        INT_t target
        DTYPE_t* o_instance
        INT_t o_target
        DTYPE_t max_dist, dist

    with nogil:
        n = idx.shape[0]
        m = dataset.shape[1]
        for i in prange(n, schedule='static', num_threads=n_jobs):
            id = idx[i]
            instance = &dataset[id, 0]
            target = targets[id]
            max_dist = INFINITY
            for j in range(n):
                o_id = idx[j]
                o_instance = &dataset[o_id, 0]
                o_target = targets[o_id]
                if o_target != target:
                    dist = c_euclidean(instance, o_instance, m)
                    if dist < max_dist:
                        max_dist = dist
            radius[id, 0] = max_dist


def _scores_radius(DTYPE_t[:, :] dataset not None, INT_t[:] targets not None, INT_t n_jobs=1):
    cdef:
        np.ndarray[DTYPE_t, ndim=2] scores = np.zeros((targets.size, 1), dtype=np.float_)
        np.ndarray[DTYPE_t, ndim=2] radius = np.zeros_like(scores)
        INT_t n, m
        INT_t i, j
        DTYPE_t denom
        DTYPE_t num
        DTYPE_t val
        DTYPE_t dist
        DTYPE_t* instance
        DTYPE_t* another_instance
        INT_t target
        INT_t another_target

    with nogil:
        n, m =  dataset.shape[0], dataset.shape[1]
        # Calculate score
        for i in prange(n, schedule='static', num_threads=n_jobs):
            num = -1
            denom = -1
            instance = &dataset[i, 0]
            target = targets[i]
            radius[i, 0] = INFINITY
            for j in range(n):
                another_instance = &dataset[j, 0]
                another_target = targets[j]
                dist = c_euclidean(instance, another_instance, m)

                val = exp(-dist)
                denom = denom + val
                if target == another_target:
                    num = num + val
                else:
                    num = num - val
                    if dist < radius[i, 0]:
                        radius[i, 0] = dist

            scores[i, 0] = num / denom
    return scores, radius
