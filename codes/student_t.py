

import numpy as np
import pandas as pd
from typing import Tuple, Dict

class StudentTConfig:
    """Configuration for Multivariate Student's t Simulation (Paper Eq. 19)"""
    def __init__(self, 
                 n_samples: int = 1000, 
                 n_star: int = 500, 
                 p: int = 2,
                 rho_pre: float = 0.9, 
                 rho_post: float = 0.0,
                 nu_pre: float = 3.0,
                 nu_post: float = 3.0):
        self.n = n_samples
        self.n_star = n_star
        self.p = p
        self.rho1 = rho_pre
        self.rho2 = rho_post
        self.nu1 = nu_pre
        self.nu2 = nu_post

def generate_student_t_segment(nu: float, rho: float, n: int, p: int) -> np.ndarray:
    # Create p-dimensional correlation matrix
    sigma = np.full((p, p), rho)
    np.fill_diagonal(sigma, 1.0)
    
    # --- ULTRA-ROBUST SAFEGUARD ---
    # 1. Use eigvalsh (guaranteed real for symmetric matrices)
    # 2. .real ensures no 'complex128' type leaks into the float64 matrix
    eigenvalues = np.linalg.eigvalsh(sigma)
    min_eig = np.min(eigenvalues).real 
    
    if min_eig < 1e-6:
        # Add a small ridge to ensure strict positive-definiteness
        # This solves the d=50 negative correlation limit mathematically
        sigma += (abs(min_eig) + 1e-5) * np.eye(p)
    # ------------------------------

    # G ~ N(0, Sigma)
    # check_valid='ignore' because we manually handled the PD check above
    G = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=n, check_valid='ignore')
    
    chi2 = np.random.chisquare(df=nu, size=(n, 1))
    scale = np.sqrt(nu / chi2)
    return scale * G

def sample_student_t_series(config: StudentTConfig) -> Tuple[np.ndarray, int]:
    """Generates the full time series with a change point as per Author's logic."""
    seg1 = generate_student_t_segment(config.nu1, config.rho1, config.n_star, config.p)
    n2 = config.n - config.n_star
    seg2 = generate_student_t_segment(config.nu2, config.rho2, n2, config.p)
    
    X = np.vstack([seg1, seg2])
    return X, config.n_star