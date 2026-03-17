

import numpy as np
from typing import Union, Tuple

def stblrnd(alpha: float, beta: float, gamma: float, delta: float, size: Union[None, Tuple[int, ...]] = None) -> np.ndarray:
    """
    Generate random samples from an alpha-stable distribution using the CMS (Chambers-Mallows-Stuck) method.
    Matches the parameterization used in the MIDAST paper (Appendix A / Eq 10-11).
    """
    if alpha <= 0 or alpha > 2 or not np.isscalar(alpha):
        raise ValueError("alpha must be a scalar which lies in the interval (0,2]")
    if abs(beta) > 1 or not np.isscalar(beta):
        raise ValueError("beta must be a scalar which lies in the interval [-1,1]")
    if gamma < 0 or not np.isscalar(gamma):
        raise ValueError("gamma must be a non-negative scalar")
    if not np.isscalar(delta):
        raise ValueError("delta must be a scalar")

    if size is None:
        size = ()
    else:
        size = tuple(size)

    # Case 1: Gaussian (Limit case alpha=2)
    if alpha == 2:
        r = np.sqrt(2) * np.random.randn(*size)

    # Case 2: Cauchy (alpha=1, beta=0)
    elif alpha == 1 and beta == 0:
        r = np.tan(np.pi / 2 * (2 * np.random.rand(*size) - 1))

    # Case 3: Levy (alpha=0.5, |beta|=1)
    elif alpha == 0.5 and abs(beta) == 1:
        r = beta / np.random.randn(*size) ** 2

    # Case 4: Symmetric Alpha-Stable (S alpha S, beta=0)
    elif beta == 0:
        V = np.pi / 2 * (2 * np.random.rand(*size) - 1)
        W = -np.log(np.random.rand(*size))
        r = (
            np.sin(alpha * V)
            / np.cos(V) ** (1 / alpha)
            * np.cos((1 - alpha) * V - np.arctan(beta * np.tan(np.pi * alpha / 2)))
            / W ** ((1 - alpha) / alpha)
        )

    # Case 5: General Case (alpha != 1)
    elif alpha != 1:
        V = np.pi / 2 * (2 * np.random.rand(*size) - 1)
        W = -np.log(np.random.rand(*size))
        const = beta * np.tan(np.pi * alpha / 2)
        B = np.arctan(const)
        S = (1 + const**2) ** (1 / (2 * alpha))
        r = (
            S
            * np.sin(alpha * V + B)
            / np.cos(V) ** (1 / alpha)
            * np.cos((1 - alpha) * V - B)
            / W ** ((1 - alpha) / alpha)
        )

    # Case 6: General Case (alpha = 1)
    else:
        V = np.pi / 2 * (2 * np.random.rand(*size) - 1)
        W = -np.log(np.random.rand(*size))
        piover2 = np.pi / 2
        sclshftV = piover2 + beta * V
        r = 1 / piover2 * (sclshftV * np.tan(V) - beta * np.log((piover2 * W * np.cos(V)) / sclshftV))

    # Affine transformation for scale and location
    if alpha != 1:
        r = gamma * r + delta
    else:
        r = gamma * r + (2 / np.pi) * beta * gamma * np.log(gamma) + delta

    return r
