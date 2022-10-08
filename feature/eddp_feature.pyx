# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: profile=True
# cython: language_level=3
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
from libc.stdlib cimport free, malloc
from libc.string cimport memset
from libc.math cimport sqrt, pow

def distance(double x1, double y1, double z1, double x2, double y2, double z2):
    cdef double s = 0.0
    s = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    return sqrt(s)

def triplet_feature(double[:, :] pos, double[:] exponents1, double[:] exponents2, double radius, double[:, :] cell, int[:] sup_cell_x, int[:] sup_cell_y, int[:] sup_cell_z):
    cdef int natoms = pos.shape[0]
    cdef int edge_cnt = 0

    cdef double s = 0.0
    cdef double* shift = <double *> malloc(24)

    cdef int n_triple = 0
    cdef int n_edge_atom = 0
    for i in range(natoms):
        n_edge_atom = 0
        for j in range(natoms):
            for sx in range(sup_cell_x[0], sup_cell_x[1]):
                for sy in range(sup_cell_y[0], sup_cell_y[1]):
                    for sz in range(sup_cell_z[0], sup_cell_z[1]):
                        memset(shift, 0, 3 * sizeof(double))
                        for ind in range(3):
                            shift[ind] = cell[0][ind] * sx + cell[1][ind] * sy + cell[2][ind] * sz
                        s = 0.0
                        for ind in range(3):
                            s = (pos[i][ind] - (pos[j][ind] + shift[ind])) ** 2
                        dist = sqrt(s)
                        if dist < radius and dist > 0.001:
                            n_edge_atom = n_edge_atom + 1
        n_triple = n_triple + n_edge_atom * (n_edge_atom - 1) / 2

    endpoint = np.zeros((200, 3))
    #features = np.zeros((n_triple, exponents1.shape[0] * exponents2.shape[0]))
    features = np.zeros((natoms, exponents1.shape[0] * exponents2.shape[0]))
    cdef int pointer = 0
    cdef double distij = 0.0
    cdef double distik = 0.0
    cdef double distjk = 0.0
    cdef double rij = 0.0
    cdef double rik = 0.0
    cdef double rjk = 0.0
    cdef int f_pointer1 = 0
    cdef int f_pointer2 = 0
    for i in range(natoms):
        endpoint = np.zeros((200, 3))
        pointer = 0
        for j in range(natoms):
            n_edge_atom = 0
            for sx in range(sup_cell_x[0], sup_cell_x[1]):
                for sy in range(sup_cell_y[0], sup_cell_y[1]):
                    for sz in range(sup_cell_z[0], sup_cell_z[1]):
                        memset(shift, 0, 3 * sizeof(double))
                        for ind in range(3):
                            shift[ind] = cell[0][ind] * sx + cell[1][ind] * sy + cell[2][ind] * sz
                        s = 0.0
                        for ind in range(3):
                            s = (pos[i][ind] - (pos[j][ind] + shift[ind])) ** 2
                        dist = sqrt(s)
                        if dist < radius and dist > 0.001:
                            for ind in range(3):
                                endpoint[pointer][ind] = pos[j][ind] + shift[ind]
                            pointer = pointer + 1
                            
        for j in range(pointer):
            for k in range(j + 1, pointer):
                distij = distance(pos[i][0], pos[i][1], pos[i][2], endpoint[j][0], endpoint[j][1], endpoint[j][2])
                distik = distance(pos[i][0], pos[i][1], pos[i][2], endpoint[k][0], endpoint[k][1], endpoint[k][2])
                distjk = distance(endpoint[j][0], endpoint[j][1], endpoint[j][2], endpoint[k][0], endpoint[k][1], endpoint[k][2])

                rij = max(2.0 * (1.0 - distij / radius), 0.0)
                rik = max(2.0 * (1.0 - distik / radius), 0.0)
                rjk = max(2.0 * (1.0 - distjk / radius), 0.0)
                f_pointer2 = 0
                for e1 in range(exponents1.shape[0]):
                    for e2 in range(exponents2.shape[0]):
                        #features[f_pointer1][f_pointer2] = pow(rij, exponents1[e1]) * pow(rik, exponents1[e1]) * pow(rjk, exponents2[e2])
                        features[i][f_pointer2] = features[i][f_pointer2] + pow(rij, exponents1[e1]) * pow(rik, exponents1[e1]) * pow(rjk, exponents2[e2])
                        f_pointer2 = f_pointer2 + 1
                f_pointer1 = f_pointer1 + 1
    return features


def twobody_feature(double[:, :] pos, const double[:] exponents, double radius, const double[:, :] cell, const int[:] sup_cell_x, const int[:] sup_cell_y, const int[:] sup_cell_z):
    cdef int natoms = pos.shape[0]
    #return sqrt(pos[0][0] ** 2)
    cdef int edge_cnt = 0

    cdef double s = 0.0
    cdef double* shift = <double *> malloc(24)

    cdef int n_edge = 0
    cdef int n_edge_atom = 0
    
    for i in range(natoms):
        n_edge_atom = 0
        for j in range(natoms):
            for sx in range(sup_cell_x[0], sup_cell_x[1]):
                for sy in range(sup_cell_y[0], sup_cell_y[1]):
                    for sz in range(sup_cell_z[0], sup_cell_z[1]):
                        memset(shift, 0, 3 * sizeof(double))
                        for ind in range(3):
                            shift[ind] = cell[0][ind] * sx + cell[1][ind] * sy + cell[2][ind] * sz
                        s = 0.0
                        for ind in range(3):
                            s = (pos[i][ind] - (pos[j][ind] + shift[ind])) ** 2
                        dist = sqrt(s)
                        if dist < radius and dist > 0.001:
                            n_edge_atom = n_edge_atom + 1
                            n_edge = n_edge + 1

    features = np.zeros((natoms, exponents.shape[0]))
    cdef int f_pointer1 = 0
    cdef double rij = 0.0
    for i in range(natoms):
        for j in range(natoms):
            n_edge_atom = 0
            for sx in range(sup_cell_x[0], sup_cell_x[1]):
                for sy in range(sup_cell_y[0], sup_cell_y[1]):
                    for sz in range(sup_cell_z[0], sup_cell_z[1]):
                        memset(shift, 0, 3 * sizeof(double))
                        for ind in range(3):
                            shift[ind] = cell[0][ind] * sx + cell[1][ind] * sy + cell[2][ind] * sz
                        s = 0.0
                        for ind in range(3):
                            s = (pos[i][ind] - (pos[j][ind] + shift[ind])) ** 2
                        dist = sqrt(s)
                    
                        if dist < radius and dist > 0.001:
                            rij = 2.0 * (1.0 - dist / radius)
                            for e in range(exponents.shape[0]):
                                #features[f_pointer1][e] = pow(rij, exponents[e])
                                features[i][e] = features[i][e] + pow(rij, exponents[e])
                            f_pointer1 = f_pointer1 + 1
    return features