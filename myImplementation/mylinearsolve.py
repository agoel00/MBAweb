import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

def mylinearsolve(X, b, n):
    def myresidual(x):
        xtop = x[:n]
        xbottom = x[n:]
        Axtop = xtop + (X@xbottom)
        Axbottom = (X.conj().T @ xtop) + xbottom
        Ax = np.concatenate((Axtop, Axbottom))
        return Ax
    myresid = LinearOperator((2*n, 2*n), matvec=myresidual)

    zeta, _ = cg(myresid, b, tol=1e-8, maxiter=100)
    alpha = zeta[:n]
    beta = zeta[n:]
    alpha, beta = alpha.reshape((n, -1)), beta.reshape((n, -1))

    return alpha, beta