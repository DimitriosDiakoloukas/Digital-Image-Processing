import numpy as np

def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    M, N, C = img_array.shape
    pixels = img_array.reshape(-1, C)            # (MN, C)
    
    # squared norms of each pixel
    sq_norms = np.sum(pixels**2, axis=1, keepdims=True)  # (MN, 1)
    
    # squared distances: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    D_sq = sq_norms + sq_norms.T - 2 * pixels.dot(pixels.T)
    D = np.sqrt(np.maximum(D_sq, 0.0))           # clamp for numerical safety
    
    affinity_mat = 1.0 / (1.0 + D)
    return affinity_mat
