# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy

cdef unsigned int UNREACHABLE_VAL = 21  #(multi_hop_max_dist = 20, so higher distance won't be reached)

def floyd_warshall(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[int, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int32)

    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long * M_ptr = &M[0, 0]
    cdef long * M_i_ptr
    cdef long * M_k_ptr

    # set unreachable nodes distance to UNREACHABLE_VAL
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = UNREACHABLE_VAL

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n * k
        for i in range(n):
            M_i_ptr = M_ptr + n * i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to UNREACHABLE_VAL
    for i in range(n):
        for j in range(n):
            if M[i][j] >= UNREACHABLE_VAL:
                path[i][j] = UNREACHABLE_VAL
                M[i][j] = UNREACHABLE_VAL

    return M, path

def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

# per node pair: returns all embeddings of all edges on the given path
def gen_edge_input(max_dist, path, edge_feat):
    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(int, order='C', casting='safe', copy=True)  #was long
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[int, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int32)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == UNREACHABLE_VAL:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k + 1], :]

    return edge_fea_all
