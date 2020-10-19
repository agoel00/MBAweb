import numpy as np 
np.set_printoptions(precision=3)

def mod_doubly_stochastic(C, maxiter=None, checkperiod=None, verbose=False):
    n = C.shape[0]
    tol = np.spacing(n)
    if not maxiter:
        maxiter = n*n
    if not checkperiod:
        checkperiod = 100

    ones_n = np.ones((n, 1))
    dd_1 = 1/np.sum(C, 0)
    dd_1 = dd_1.reshape(1, -1)
    d_2 = 1/(C @ dd_1.T)
    d_2_prev = d_2
    it = 0
    gap = np.Inf
    if np.any(np.isinf(d_2)) or np.any(np.isnan(d_2)):
        print("Nan or Inf occurred. DS projection iter: {}, error: {}\n".format(it,gap))

    while it < maxiter:
        if verbose:
            print("Iteration: {}\n".format(it))
        it = it+1

        d_2 = ones_n/(C @ (ones_n/(C.conj().T @ d_2)))

        if np.any(np.isinf(d_2)) or np.any(np.isnan(d_2)):
            print("Nan or Inf occurred. DS projection iter: {}, error: {}\n".format(it,gap))
            d_2 = d_2_prev
            break

        if np.mod(it, checkperiod) == 0:
            row = d_2.conj().T @ C
            d_1 = ones_n.conj().T / row
            d_2 = 1/(C @ d_1.T)
            row = d_2.conj().T @ C
            gap = np.amax(abs(row * d_1 -1), 0)
            if np.any(np.isnan(gap)):
                break
            if np.any(gap <= tol):
                break

        
        d_2_prev = d_2

    print("DS Projection iter: {}, error: {}\n".format(it, np.sum(gap)))
    d_1 = ones_n/(C.conj().T @ d_2)
    B = (d_2 @ d_1.conj().T)*C

    return B