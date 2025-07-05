import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    n = affinity_mat.shape[0]
    if affinity_mat.shape[1] != n:
        raise ValueError("affinity_mat must be square")
    if k < 2 or k >= n:
        raise ValueError("k must satisfy 2 <= k < n")

    # Build sparse W (zero self‐loops) and D
    W = csr_matrix(affinity_mat)
    W.setdiag(0)
    degrees = np.ravel(W.sum(axis=1))
    D = diags(degrees)

    # Form Laplacian
    L = D - W

    # Solve generalized eigenproblem L v = λ D v for k smallest λ
    vals, vecs = eigs(L, M=D, k=k, which='SM')  

    # for deterministic results, use a fixed nonzero vector as v0 uncomment this ARPACK’s
    # internal random‐start for the Lanczos iteration
    # n = L.shape[0]
    # v0 = np.ones(n)       
    # vals, vecs = eigs(L, M=D, k=k, which='SM', v0=v0)
    U = np.real(vecs)      # shape (n, k)

    # Cluster rows of U
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(U)
    return km.labels_.astype(float)


def calculate_n_cut_value(affinity_mat: np.ndarray,
                          cluster_idx: np.ndarray) -> float:
    labels = cluster_idx.astype(int)
    if set(np.unique(labels)) - {0,1}:
        raise ValueError("cluster_idx must contain exactly two labels 0 and 1")

    W = affinity_mat
    n = W.shape[0]

    # indices of the two clusters
    A = np.where(labels == 0)[0]
    B = np.where(labels == 1)[0]

    # assoc(A, V) = sum of W[u,t] for u in A, t in all nodes
    assoc_A_V = W[A, :].sum()
    assoc_B_V = W[B, :].sum()

    # assoc(A, A) and assoc(B, B)
    assoc_A_A = W[np.ix_(A, A)].sum()
    assoc_B_B = W[np.ix_(B, B)].sum()

    nassoc = (assoc_A_A / assoc_A_V) + (assoc_B_B / assoc_B_V)
    ncut_value = 2.0 - nassoc
    return float(ncut_value)


def n_cuts_recursive(affinity_mat: np.ndarray,
                     T1: int,
                     T2: float) -> np.ndarray:
    n = affinity_mat.shape[0]
    if affinity_mat.shape[1] != n:
        raise ValueError("affinity_mat must be square")

    # Initialize queue of clusters to process: each entry is an array of node indices
    queue = [np.arange(n)]
    final_clusters = []

    while queue:
        idx = queue.pop(0)
        # Stop if cluster too small
        if idx.size <= T1:
            final_clusters.append(idx)
            continue

        # Extract subgraph affinity
        W_sub = affinity_mat[np.ix_(idx, idx)]
        # Bisect with n_cuts(k=2)
        labels_sub = n_cuts(W_sub, k=2)
        # Compute its Ncut value
        ncut_val = calculate_n_cut_value(W_sub, labels_sub)

        # Decide to split or not
        if ncut_val >= T2:
            # don’t split further
            final_clusters.append(idx)
        else:
            # split into two parts and re‐queue
            part0 = idx[labels_sub.astype(int) == 0]
            part1 = idx[labels_sub.astype(int) == 1]
            queue.append(part0)
            queue.append(part1)

    # Build final label vector
    cluster_idx = np.empty(n, dtype=float)
    for label, members in enumerate(final_clusters):
        cluster_idx[members] = float(label)

    return cluster_idx
