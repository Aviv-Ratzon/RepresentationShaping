from itertools import product
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch

def cosine_similarity(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    else:
        raise TypeError("Inputs must be both numpy arrays or both torch tensors")

def vector_angle(a, b):
    return np.rad2deg(np.arccos(cosine_similarity(a, b)))

def get_r_2(X, y):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return r2_score(y, y_pred)

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

def get_upper_triangle(matrix):
    """
    Get the upper triangle of a square matrix.
    """
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]

import numpy as np
from scipy.linalg import svd
from scipy.linalg import subspace_angles

def compute_covariance(X):
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    return cov

def principal_directions(cov, explained_var=0.95):
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    explained_variance_ratio = np.cumsum(eigvals) / np.sum(eigvals)
    k = np.where(explained_variance_ratio > explained_var)[0][0] if explained_variance_ratio[0] < explained_var else 0
    print(k)
    return eigvecs[:, :k+1]

def alignment_score(blob1, blob2, explained_var=0.95):
    # Compute covariance matrices
    cov1 = compute_covariance(blob1)
    cov2 = compute_covariance(blob2)

    # Get top-k principal directions
    dirs1 = principal_directions(cov1, explained_var)
    dirs2 = principal_directions(cov2, explained_var)

    # Compute canonical (principal) angles
    angles = subspace_angles(dirs1, dirs2)

    # Alignment score: cosine of smallest angle (closer to 1 means more aligned)
    score = np.cos(angles[0])
    return score


# def calc_NC1(h, y):
#     with torch.no_grad():
#         classes = torch.unique(y.argmax(1))
#         Sw = 0
#         mean_c = []
#         for c in classes:
#             h_c = h[y.argmax(1) == c]
#             z = h_c - h_c.mean(0).unsqueeze(0)
#             cov = z.unsqueeze(-1) @ z.unsqueeze(1)
#             Sw += cov.sum(0)
#             mean_c.append(h_c.mean(0))
#         Sw /= y.shape[0]
#         M = torch.stack(mean_c).T
#         muG = torch.mean(M, dim=1, keepdim=True)  # CHW 1
#         M_ = M - muG
#         Sb = torch.matmul(M_, M_.T) / len(classes)
#     Sw = Sw.cpu().numpy()
#     Sb = Sb.cpu().numpy()
#     eigvec, eigval, _ = svds(Sb, k=len(classes) - 1)
#     inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
#     return np.trace(Sw @ inv_Sb)

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def calc_NC1(h, y):
    """
    Compute NC1 from hidden activations h and labels y.

    Parameters:
    - h: ndarray of shape (n_samples, hidden_dim), hidden layer activations
    - y: ndarray of shape (n_samples,), class labels

    Returns:
    - nc1: scalar, NC1 value
    """
    h = np.array(h)
    y = np.array(y)
    classes = np.unique(y)
    num_classes = len(classes)
    hidden_dim = h.shape[1]

    # Compute class means and global mean
    class_means = np.zeros((num_classes, hidden_dim))
    global_mean = np.mean(h, axis=0)

    Sw = np.zeros((hidden_dim, hidden_dim))  # within-class covariance
    Sb = np.zeros((hidden_dim, hidden_dim))  # between-class covariance

    for i, cls in enumerate(classes):
        h_cls = h[y == cls]
        mu_cls = np.mean(h_cls, axis=0)
        class_means[i] = mu_cls

        # Within-class scatter
        centered = h_cls - mu_cls
        Sw += centered.T @ centered

    # Between-class scatter
    for i, mu_cls in enumerate(class_means):
        n_i = np.sum(y == classes[i])
        diff = (mu_cls - global_mean).reshape(-1, 1)
        Sb += n_i * (diff @ diff.T)

    trace_Sw = np.trace(Sw)
    trace_Sb = np.trace(Sb)

    nc1 = trace_Sw / trace_Sb if trace_Sb != 0 else np.inf
    return nc1

def calc_PR(h):
    cov = compute_covariance(h)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)


# PCA function using SVD
def pca_torch(X, k=None):
    # Center the data
    X_centered = X - X.mean(dim=0, keepdim=True)

    # SVD decomposition
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # Select top-k components if specified
    if k is not None:
        S = S[:k]
        Vt = Vt[:k]

    # Compute explained variance
    n_samples = X.shape[0]
    explained_variance = (S ** 2) / (n_samples - 1)

    # Total variance is the sum of variances of all features
    total_var = X_centered.var(dim=0, unbiased=True).sum()
    explained_variance_ratio = explained_variance / total_var

    return {
        'components': Vt,  # shape: (k, n_features)
        'explained_variance': explained_variance.cpu().numpy(),  # shape: (k,)
        'explained_variance_ratio': explained_variance_ratio.cpu().numpy()  # shape: (k,)
    }


def get_all_key_combinations(my_dict):
    # Get keys and lists of values
    keys = list(my_dict.keys())
    values_lists = [my_dict[key] for key in keys]

    # Iterate over all combinations
    combinations = []
    for combination in product(*values_lists):
        current = dict(zip(keys, combination))
        combinations.append(current)  # current is a dict: {key: value, ...}
    return combinations

def factorize_matrix(M, N=None):
    D1, D2 = M.shape
    # Compute full SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rank = np.sum(S > 1e-10)  # numerical rank

    if N is None:
        N = min(D1,D2)  # default N to rank(M)
    
    # Handle case where N > min(D1,D2)
    if N > min(D1,D2):
        # Pad U with random orthonormal columns
        U_extra = np.random.randn(D1, N - min(D1,D2))
        U_extra, _ = np.linalg.qr(U_extra)
        U = np.hstack([U, U_extra])
        
        # Pad S with zeros
        S = np.pad(S, (0, N - len(S)), mode='constant')
        
        # Pad Vt with random orthonormal rows
        Vt_extra = np.random.randn(N - min(D1,D2), D2) 
        Vt_extra, _ = np.linalg.qr(Vt_extra.T)
        Vt_extra = Vt_extra.T
        Vt = np.vstack([Vt, Vt_extra])

    # Take N components
    U_N = U[:, :N]
    S_N = np.diag(S[:N])
    Vt_N = Vt[:N, :]

    # Generate random orthogonal matrix
    Q = np.random.randn(N, N)
    Q, _ = np.linalg.qr(Q)
    
    # Create random factorization that still reconstructs M
    A = U_N @ np.sqrt(S_N) @ Q
    B = Q.T @ np.sqrt(S_N) @ Vt_N
    return A, B