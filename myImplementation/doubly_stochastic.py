import numpy as np
from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold, Manifold
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp
from mod_doubly_stochastic import mod_doubly_stochastic
from mylinearsolve import mylinearsolve
from uf import uf

class DoublyStochastic(EuclideanEmbeddedSubmanifold):
    """
    Manifold of n-by-n doubly stochastic matrices with positive entries.

    This is a Pymanopt manifold structure to optimize over the set of n-by-n
    matrices with (strictly) positive entries and such that the entries of
    each column and each row sum to one.

    Points on the manifold and tangent vectors are represented naturally as 
    symmetric matrices of size n. The Riemannian metric imposed on the manifold
    is the Fisher metric, that is, if X is a point on the manifold and U, V
    are two tangent vectors:
        inner(X, U, V) = <U, V>_X = sum(sum(U*V/X))
    
    """
    def __init__(self, n):
        self._n = n 
        name = "{}x{} doubly stochastic matrices with positive entries".format(self._n, self._n)
        dimension = (n-1)**2
        self.e = np.ones((n, 1))
        self.DSmaxiter = min(2*n, 1000)
        super().__init__(name, dimension)

    def inner(self, X, eta, zeta):
        return np.sum((eta.flatten('F') * zeta.flatten('F'))/X.flatten('F'))
    
    def norm(self, X, eta):
        return np.sqrt(self.inner(X, eta, eta))
    
    def dist(self, X, Y):
        return NotImplementedError
    
    def typicaldist(self):
        return self._n
    
    def rand(self):
        """
        Pick a random point on the manifold.
        """
        # Random point in the ambient space.
        Z = abs(np.random.randn(self._n, self._n))
        # Projection onto the manifold
        X = mod_doubly_stochastic(Z, self.DSmaxiter)
        return X 
    
    def randvec(self, X):
        """
        Pick a random vector in the tangent space at X.
        """
        # A random vector in the ambient space.
        Z = np.random.randn(self._n, self._n)
        # Projection of the vector onto the tangent space.
        b = np.concatenate((np.sum(Z, 1), np.sum(Z, 0).conj().T))
        alpha, beta = mylinearsolve(X, b, self._n)
        eta = Z - (alpha @ self.e.conj().T + self.e @ beta.conj().T) * X
        # Normalizing the vector
        nrm = self.norm(X, eta)
        eta = eta / nrm 
        return eta

    def proj(self, X, eta):
        """
        Projection of vector eta in the ambient space to the tangent space.
        """
        b = np.concatenate((np.sum(eta, 1), np.sum(eta, 0).conj().T))
        alpha, beta = mylinearsolve(X, b, self._n)
        etaproj = eta - (alpha @ self.e.conj().T + self.e @ beta.conj().T) * X
        return etaproj

    def egrad2rgrad(self, X, egrad):
        """
        Conversion of Euclidean to Riemannian gradient.
        """
        mu = X * egrad
        b = np.concatenate((np.sum(mu, 1), np.sum(mu, 0).conj().T))
        alpha, beta = mylinearsolve(X, b, self._n)
        rgrad = mu - (alpha @ self.e.conj().T + self.e @ beta.conj().T) * X
        return rgrad

    def ehess2ress(self, X, egrad, ehess, eta):
        """
        Conversion of Euclidean to Riemannian Hessian.
        """
        # Computing the directional derivative of the Riemanniaan gradient.
        gamma = egrad * X
        gammadot = ehess*X + egrad*eta

        A = np.block([[np.eye(self._n), X], [X.conj().T, np.eye(self._n)]])
        b = np.concatenate((np.sum(gamma, 1), np.sum(gamma, 0).conj().T))
        bdot = np.concatenate((np.sum(gammadot, 1), np.sum(gammadot, 0).conj().T))

        alpha, beta = mylinearsolve(X, b, self._n)
        alphadot, betadot = mylinearsolve(X, bdot - np.concatenate((eta@beta), (eta.conj().T@alpha)), self._n)
        
        S = (alpha @ self.e.conj().T + self.e @ beta.conj().T)
        deltadot = gammadot - (alphadot @ self.e.conj().T + self.e @ betadot.conj().T)*X - S*eta
       
        # Projecting gamma
        delta = gamma - S*X

        # Computing and projecting nabla
        nabla = deltadot - 0.5@(delta*eta)/X
        rhess = self.proj(X, nabla)

        return rhess

    def exp(self, X, eta, t=1.0):
        """
        First order retraction.
        """
        Y = X * np.exp(t*(eta/X))
        Y = mod_doubly_stochastic(Y, self.DSmaxiter)
        Y = np.maximum(Y, np.spacing(1))
        return Y

    retr = exp

    def zerovec(self, X):
        return np.zeros((self._n, self._n))

    def transp(self, X1, X2, d):
        return self.proj(X2, d)

    def vec(self, X, U):
        return U.flatten()