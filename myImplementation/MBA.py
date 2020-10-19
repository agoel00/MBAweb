import torch 
import numpy as np
import pymanopt
from pymanopt.solvers import ConjugateGradient
from doubly_stochastic import DoublyStochastic
import pickle
from uf import uf

import networkx as nx
from karateclub import NetMF

np.set_printoptions(precision=3)

# X = np.random.randn(100, 10)
# # X = np.random.randint(1, 20, size=(1000, 10)).astype(float)
# Z = np.random.permutation(X)
# X = torch.from_numpy(X)
# Z = torch.from_numpy(Z)

print("Creating Graph Embedddings X and Z...\n")
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
P = np.random.permutation(np.eye(A.shape[0], dtype=np.int))
_A = (P @ A) @ (np.linalg.inv(P))
_G = nx.from_numpy_matrix(_A)

correspondences = {}
for val in range(P.shape[0]):
    correspondences[np.argmax(P[val, :])] = val

model = NetMF(dimensions=10)
_model = NetMF(dimensions=10)
model.fit(G)
_model.fit(_G)
X = model.get_embedding()
Z = model.get_embedding()
print("Graph Embeddings Created...\n")
X,Z = torch.from_numpy(X).double(), torch.from_numpy(Z).double()

N = X.shape[0]

BtB = torch.matmul(Z.T, Z)
DtD = torch.matmul(X.T, X)
normBtB = torch.norm(BtB.flatten())
normDtD = torch.norm(DtD.flatten())

manifold = DoublyStochastic(N)
regularizer = 0

@pymanopt.function.PyTorch
def cost(Y):
    A = torch.matmul(Y.T, X)
    AtA = torch.matmul(A.T, A)
    C = torch.matmul(Y, Z)
    CtC = torch.matmul(C.T, C)
    XtYZ = torch.matmul(X.T, C)
    BtA = XtYZ.T 
    DtC = XtYZ

    f = 0.5 * (
        torch.norm(AtA.flatten())**2 + normBtB**2
    ) - torch.norm(BtA.flatten())**2

    f += 0.5 * (
        torch.norm(CtC.flatten())**2 + normDtD**2
    ) - torch.norm(DtC.flatten())**2

    f += 0.5*regularizer*torch.norm(Y, 'fro')**2
    return f

problem = pymanopt.Problem(manifold, cost=cost)
solver = ConjugateGradient(maxiter=200)
print("Starting optimization...\n")
Yopt = solver.solve(problem)
print("Optimization finished\n")
print(Yopt.sum(0))
print(Yopt.sum(1))

X, Z = X.numpy(), Z.numpy()

W = uf(X.T @ (Yopt @ Z))

# Alignment of rows / Node correspondences
YZ = Yopt@Z
YX = Yopt.T @ X

print("||X - YZ||_fro", np.linalg.norm((X - YZ), 'fro'))
print("||Z - YX||_fro", np.linalg.norm((Z - YX), 'fro'))

# embedding space transformation
# Orthogonal Procrusts
# |XW - YZ| -> 0
XW = X@W 
print("||XW - YZ||_fro", np.linalg.norm((XW - YZ), 'fro'))

np.save('X.npy', X)
np.save('Z.npy', Z)
np.save('Yopt.npy', Yopt)
np.save('W.npy', W)
with open('true_align.pickle', 'wb') as f:
    pickle.dump(correspondences, f)