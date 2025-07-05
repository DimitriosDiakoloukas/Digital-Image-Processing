import numpy as np

def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    """
    Convert an image to a graph representation where each pixel is a node
    and edges are defined by the Euclidean distance between pixels.
    Parameters:
    img_array (np.ndarray): Input image as a 3D numpy array of shape (M, N, C)
                            where M is height, N is width, and C is color channels.

    Returns:
    np.ndarray: Affinity matrix representing the graph, where each entry
                corresponds to the affinity between two pixels.
    """
    M, N, C = img_array.shape
    pixels = img_array.reshape(-1, C)            # (MN, C)
    
    # squared norms of each pixel
    sq_norms = np.sum(pixels**2, axis=1, keepdims=True)  # (MN, 1)
    
    # squared distances: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    D_sq = sq_norms + sq_norms.T - 2 * pixels.dot(pixels.T)
    D = np.sqrt(np.maximum(D_sq, 0.0))           # clamp for numerical safety
    
    affinity_mat = np.exp(-D)
    return affinity_mat
