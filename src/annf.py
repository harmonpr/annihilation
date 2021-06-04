import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def distance_each_particles(n, x):
    r_ij = np.array([x]).T - x + np.eye(n)
    return r_ij

def annihilation_rule(n, b, x, d):
    b_i = b.copy()
    for i in range(n-1):
        if b_i[i] != 0:
            j = i+1
            while(j <= n-1) and (b_i[j] == 0):
                j += 1
            if j <= n-1 and (b_i[i]*b_i[j] == -1) and (np.abs(x[i]-x[j]) <= d):
                b_i[i] = 0
                b_i[j] = 0
                i = j+1
            else:
                i = j
    return b_i

def force(n, b, r):
    b_ij = np.array([b]).T * b
    x_ij = (1/r) - np.eye(n)
    f_ij = ((x_ij*b_ij).sum(axis=1)) / n
    return f_ij

def x_b_f_calculation(x, b, f):
    id_nonZero = np.array(np.nonzero(b))
    x_cl = x[id_nonZero]
    b_cl = b[id_nonZero]
    f_cl = f[id_nonZero]
    return x_cl, b_cl, f_cl

def d_min_plus(x, b, d_min, d_plus):
    xd_min = np.ones(len(b[0]))
    xd_plus = np.ones(len(b[0]))
    for i in range(len(b[0]) - 1):
        if (b[0,i] + b[0,i+1]) == -2:
            xd_m = np.abs(x[0,i] - x[0,i+1])
            xd_min[i] = xd_m
        if (b[0,i] + b[0,i+1]) == 2:
            xd_p = np.abs(x[0,i] - x[0,i+1])
            xd_plus[i] = xd_p    
    if not xd_min.tolist():
        xd_min = d_min
    if not xd_plus.tolist():
        xd_plus = d_plus
    return xd_min, xd_plus

def first_stability_condition(x, b, f):
    fi_1 = np.zeros(len(f[0]))
    xi_1 = np.zeros(len(f[0]))
    for i in range(len(b[0]) - 1):
        if (b[0,i] * b[0,i+1]) == -1:
            fi_1[i] = f[0,i] - f[0,i+1]
            xi_1[i] = x[0,i+1] - x[0,i]
    h_1 = (xi_1[fi_1 > 0]) / fi_1[fi_1 > 0]
    return h_1

def second_stability_condition(x, b, f, d_plus):
    fi_2 = np.zeros(len(f[0]))
    xi_2 = np.zeros(len(f[0]))
    for i in range(len(b[0]) - 1):
        if (b[0,i] * b[0,i+1]) == 2:
            fi_2[i] = f[0,i+1] - f[0,i]
            xi_2[i] = x[0,i+1] - x[0,i]
    h_2 = (d_plus - xi_2[fi_2 < 0]) / fi_2[fi_2 < 0] 
    return h_2

def third_stability_condition(x, b, f, d_min):
    fi_3 = np.zeros(len(f[0]))
    xi_3 = np.zeros(len(f[0]))
    for i in range(len(b[0]) - 1):
        if (b[0,i] * b[0,i+1]) == -2:
            fi_3[i] = f[0,i+1] - f[0,i]
            xi_3[i] = x[0,i+1] - x[0,i]
    h_3 = (d_min - xi_3[fi_3 < 0]) / fi_3[fi_3 < 0] 
    return h_3
