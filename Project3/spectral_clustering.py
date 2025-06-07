import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    n = affinity_mat.shape[0]
    if affinity_mat.shape[1] != n:
        raise ValueError("affinity_mat must be square")
    if k >= n:
        raise ValueError("k must be strictly less than the number of nodes")

    # sparse W, D
    W = csr_matrix(affinity_mat)
    W.setdiag(0)    # optional: zero self-loops
    degrees = np.ravel(W.sum(axis=1))
    D = diags(degrees)

    # Laplacian
    L = D - W

    # k smallest eigenvectors
    vals, vecs = eigs(L, k=k, which='SM')
    U = np.real(vecs)

    # unsupervised k-means
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(U)
    return km.labels_.astype(float)
