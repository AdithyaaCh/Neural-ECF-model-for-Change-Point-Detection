import numpy as np
from trajectory_utils import stblrnd

class SubGaussianConfig:
    """Configuration for Sub-Gaussian Alpha-Stable Simulation (Paper Eq. 14)"""
    def __init__(self, 
                 n_samples: int = 1000, 
                 n_star: int = 500, 
                 p: int = 2,
                 rho_pre: float = 0.9, 
                 rho_post: float = 0.0,
                 alpha_pre: float = 1.8,
                 alpha_post: float = 1.8):
        self.n = n_samples
        self.n_star = n_star
        self.p = p
        self.rho1 = rho_pre
        self.rho2 = rho_post
        self.alpha1 = alpha_pre
        self.alpha2 = alpha_post

def generate_subgaussian_segment(alpha: float, rho: float, n: int, p: int) -> np.ndarray:
    """
    Generate a segment of Sub-Gaussian alpha-stable vectors.
    X = A^{1/2} * G
    """
    sigma = np.full((p, p), rho)
    np.fill_diagonal(sigma, 1.0)
    
    # Gaussian base G ~ N(0, Sigma)
    G = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=n)
    
    # Mixing variable A ~ S_{alpha/2}(gamma, 1, 0)
    # gamma parameter from Paper Eq. 11: (cos(pi * alpha / 4))^(2/alpha)
    gamma_val = (np.cos(np.pi * alpha / 4.0)) ** (2.0 / alpha)
    
    A = stblrnd(
        alpha=alpha / 2.0,
        beta=1.0,
        gamma=gamma_val,
        delta=0.0,
        size=(n, 1)
    )
    
    # Mixing: X = sqrt(A) * G
    return np.sqrt(np.abs(A)) * G

def sample_subgaussian_series(config: SubGaussianConfig) -> tuple[np.ndarray, int]:
    """Generates the full time series mirroring the author's v2 simulation."""
    seg1 = generate_subgaussian_segment(config.alpha1, config.rho1, config.n_star, config.p)
    n2 = config.n - config.n_star
    seg2 = generate_subgaussian_segment(config.alpha2, config.rho2, n2, config.p)
    
    X = np.vstack([seg1, seg2])
    return X, config.n_star