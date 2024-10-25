import numpy as np
import torch

def sliced_wasserstein(X, Y, centering=True):
    """
    Approximates SW with Wasserstein distance between Gaussian approximations
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.float, device=device)
    d = X.shape[1]
    if centering:
        # Center the data
        mean_X = torch.mean(X, dim=0)
        mean_Y = torch.mean(Y, dim=0)
        X = X - mean_X
        Y = Y - mean_Y
    # Approximate SW
    m2_Xc = torch.mean(torch.linalg.norm(X, dim=1) ** 2) / d
    m2_Yc = torch.mean(torch.linalg.norm(Y, dim=1) ** 2) / d
    sw = torch.abs(m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2))
    return float(sw)

def montecarlo_sw(X, Y, L=100, p=2):
    """
    Computes the Monte Carlo estimation of Sliced-Wasserstein distance between empirical distributions
    """
    N, d = X.shape
    order = p
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.float, device=device)
    # Project data
    theta = torch.randn(L, d)
    theta = theta / torch.linalg.norm(theta, dim=1)[:, None]  # normalization (theta is in the unit sphere)
    theta = torch.t(theta)
    xproj = torch.matmul(X, theta)
    yproj = torch.matmul(Y, theta)
    # Sort projected data
    xqf, _ = torch.sort(xproj, dim=0)
    yqf, _ = torch.sort(yproj, dim=0)
    # Compute expected SW distance
    sw_dist = torch.mean(torch.abs(xqf - yqf) ** order)
    sw_dist = sw_dist ** (1/order)
    return sw_dist

