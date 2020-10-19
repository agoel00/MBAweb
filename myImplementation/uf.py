import numpy as np
from scipy.linalg import svd

def uf(A):
    threshold = 1e-8
    u, s, v = svd(A)
    u = u[:, :s.shape[0]]
    v = v.T
    v = v[:, :s.shape[0]]
    nonzeroS = s > threshold
    A1 = u[:, nonzeroS] @ (v[:, nonzeroS]).T
    return A1