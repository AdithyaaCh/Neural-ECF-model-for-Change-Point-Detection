
from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from common.algorithm12 import algorithm_1_window, algorithm_2_k, k_rule_of_thumb
from vendored_midast import ChangeDetector


def _select_w_k(
    X: np.ndarray,
    dim: int,
    test_name: str,
    *,
    fast: bool,
    dist: str = "student_t",
    target_cps: Optional[int] = None,
) -> tuple[int, int]:
    """Pick window w via Algorithm 1 and grouping factor k via Algorithm 2.

    `fast=True` skips the Monte-Carlo loops and uses the paper's rule-of-thumb
    `k = w/(100*s)` with a length-aware default w. This is intended only for
    quick smoke-tests; production runs go through the full Algorithm 1/2 path.
    """
    n_rows = X.shape[0]
    s = 10  # MIDAST shift default

    if fast:
        target_cps = target_cps or 1
        w = min(200, max(50, n_rows // (2 * (target_cps + 1))))
        k = k_rule_of_thumb(w, s)
        return int(w), int(k)

    rho_pre, rho_post = 0.5, 0.0
    if dist == "student_t":
        tail_pre, tail_post = 3.0, 8.0
    else:
        tail_pre, tail_post = 1.95, 1.6

    w = algorithm_1_window(
        test_name=test_name, dist=dist, dim=dim,
        rho_pre=rho_pre, rho_post=rho_post,
        tail_pre=tail_pre, tail_post=tail_post,
        M=200, target_power=0.9, c=0.05,
    )
    k = algorithm_2_k(
        test_name=test_name, dist=dist, dim=dim, w=w, s=s,
        rho_pre=rho_pre, rho_post=rho_post,
        tail_pre=tail_pre, tail_post=tail_post,
        M=80, tol=max(20, w // 4), c=0.05,
    )
    return int(w), int(k)


def _run_midast(
    X: np.ndarray,
    dim: int,
    test_name: str,
    target_cps: Optional[int],
    *,
    fast: bool,
    dist: str,
) -> np.ndarray:
    w, k = _select_w_k(X, dim, test_name, fast=fast, dist=dist, target_cps=target_cps)
    s = 10
    shift_group = max(1, int(k * w / 100 * s))

    det = ChangeDetector(test_name=test_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_df = det.fit(X, window_size=w, shift=s)
    cps = det.analyze_results(
        res_df,
        output_type="np.array",
        alpha=0.05,
        shift_group=shift_group,
        max_no_changes=target_cps,
        based_on="statistic",  
    )
    if cps is None:
        return np.array([], dtype=int)
    return np.sort(np.asarray(cps).astype(int))


def run_midast_ks(X, dim, target_cps=None, *, fast=False, dist="student_t"):
    return _run_midast(X, dim, "KSTest", target_cps, fast=fast, dist=dist)


def run_midast_ks_dkw(X, dim, target_cps=None, *, fast=False, dist="student_t"):
    return _run_midast(X, dim, "KSTest_DKW", target_cps, fast=fast, dist=dist)


def run_midast_mmd_python(X, dim, target_cps=None, *, fast=False, dist="student_t"):

    w, k = _select_w_k(X, dim, "MMDTest", fast=fast, dist=dist, target_cps=target_cps)
    s = 10
    shift_group = max(1, int(k * w / 100 * s))

    from sklearn.metrics.pairwise import rbf_kernel

    n_rows = X.shape[0]
    rows = []
    rng = np.random.default_rng(0)
    for ind in range(0, n_rows - 2 * w, s):
        v1 = X[ind : ind + w]
        v2 = X[ind + w : ind + 2 * w]
        gamma = 1.0
        XX = rbf_kernel(v1, v1, gamma)
        YY = rbf_kernel(v2, v2, gamma)
        XY = rbf_kernel(v1, v2, gamma)
        nA, nB = v1.shape[0], v2.shape[0]
        true_stat = (
            (XX.sum() - np.trace(XX)) / (nA * (nA - 1))
            + (YY.sum() - np.trace(YY)) / (nB * (nB - 1))
            - 2 * XY.sum() / (nA * nB)
        )
        ge = 0
        n_perm = 49
        pooled = np.vstack([v1, v2])
        for _ in range(n_perm):
            rng.shuffle(pooled)
            a, b = pooled[:nA], pooled[nA:]
            AA = rbf_kernel(a, a, gamma)
            BB = rbf_kernel(b, b, gamma)
            AB = rbf_kernel(a, b, gamma)
            fake = (
                (AA.sum() - np.trace(AA)) / (nA * (nA - 1))
                + (BB.sum() - np.trace(BB)) / (nB * (nB - 1))
                - 2 * AB.sum() / (nA * nB)
            )
            if fake >= true_stat:
                ge += 1
        pval = (ge + 1) / (n_perm + 1)
        rows.append({"id": ind + w, "window1_start": ind, "window2_end": ind + 2 * w,
                     "statistic": float(true_stat), "pvalue": float(pval)})
    res_df = pd.DataFrame(rows)
    det = ChangeDetector()
    det.window_size = w
    cps = det.analyze_results(
        res_df, output_type="np.array", alpha=0.05,
        shift_group=shift_group, max_no_changes=target_cps, based_on="statistic",
    )
    if cps is None:
        return np.array([], dtype=int)
    return np.sort(np.asarray(cps).astype(int))


def run_midast_mmd_r(X, dim, target_cps=None, *, fast=False, dist="student_t"):
    if not os.getenv("USE_R_TESTS"):
        raise RuntimeError("USE_R_TESTS=1 not set; R-backed MMD unavailable")
    return _run_midast(X, dim, "MMDTest", target_cps, fast=fast, dist=dist)


def run_midast_kernel_r(X, dim, target_cps=None, *, fast=False, dist="student_t"):
    if not os.getenv("USE_R_TESTS"):
        raise RuntimeError("USE_R_TESTS=1 not set; R-backed Kernel test unavailable")
    return _run_midast(X, dim, "KernelDensitiesTest", target_cps, fast=fast, dist=dist)


def run_midast_cramer_r(X, dim, target_cps=None, *, fast=False, dist="student_t"):
    if not os.getenv("USE_R_TESTS"):
        raise RuntimeError("USE_R_TESTS=1 not set; R-backed Cramer test unavailable")
    return _run_midast(X, dim, "CramerTest", target_cps, fast=fast, dist=dist)


def run_midast_copula_r(X, dim, target_cps=None, *, fast=False, dist="student_t"):
    if not os.getenv("USE_R_TESTS"):
        raise RuntimeError("USE_R_TESTS=1 not set; R-backed Copula test unavailable")
    return _run_midast(X, dim, "CopulaTest", target_cps, fast=fast, dist=dist)


def midast_mmd_dispatch(X, dim, target_cps=None, *, fast=False, dist="student_t"):
    """Use R-backed MMD if available, else pure-Python permutation MMD."""
    if os.getenv("USE_R_TESTS"):
        return run_midast_mmd_r(X, dim, target_cps, fast=fast, dist=dist)
    return run_midast_mmd_python(X, dim, target_cps, fast=fast, dist=dist)
