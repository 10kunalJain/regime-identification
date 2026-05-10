"""Regime-label alignment: Hungarian assignment minimizing W₂ between Gaussian regimes.

For Gaussian regimes (μ, Σ), the squared 2-Wasserstein distance has a closed
form:

  W₂²(N(μ_a, Σ_a), N(μ_b, Σ_b))
    = ||μ_a - μ_b||² + tr(Σ_a + Σ_b - 2(Σ_a^{1/2} Σ_b Σ_a^{1/2})^{1/2})

We use this as the cost matrix for `scipy.optimize.linear_sum_assignment`. The
returned permutation `P` says regime `b[P[k]]` aligns with reference regime
`a[k]`. Combined with a SPY-mean-rank tiebreaker (per the Q3 lock), this
removes label-switching across folds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment


def w2_squared_gaussian(
    mu_a: np.ndarray, cov_a: np.ndarray, mu_b: np.ndarray, cov_b: np.ndarray
) -> float:
    """Squared 2-Wasserstein distance between two multivariate Gaussians."""
    mu_diff = mu_a - mu_b
    mean_term = float(mu_diff @ mu_diff)

    sqrt_a = np.asarray(sqrtm(cov_a))
    inner = sqrt_a @ cov_b @ sqrt_a
    sqrt_inner = np.asarray(sqrtm(inner))
    cov_term = float(np.real(np.trace(cov_a + cov_b - 2.0 * sqrt_inner)))

    return mean_term + max(cov_term, 0.0)


def hungarian_align(
    means_a: np.ndarray,
    covs_a: np.ndarray,
    means_b: np.ndarray,
    covs_b: np.ndarray,
) -> np.ndarray:
    """Hungarian assignment minimizing W₂² between regime sets a and b.

    Args:
        means_a: shape (K, D) reference means.
        covs_a: shape (K, D, D) reference covariances.
        means_b: shape (K, D) candidate means.
        covs_b: shape (K, D, D) candidate covariances.

    Returns:
        Permutation `P` of length K such that candidate regime `P[k]` aligns
        with reference regime `k`. Equivalent to `col_ind` from
        `linear_sum_assignment`.
    """
    K = means_a.shape[0]
    if not (means_b.shape[0] == K and covs_a.shape[0] == K and covs_b.shape[0] == K):
        raise ValueError("a and b must have the same number of regimes")

    cost = np.empty((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            cost[i, j] = w2_squared_gaussian(means_a[i], covs_a[i], means_b[j], covs_b[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind is 0..K-1 in order; col_ind gives the assigned column for each row.
    if not np.array_equal(row_ind, np.arange(K)):
        # Rearrange so row_ind is identity (linear_sum_assignment doesn't guarantee this
        # when costs are degenerate, though in practice it does for square matrices).
        order = np.argsort(row_ind)
        col_ind = col_ind[order]
    return col_ind


def align_by_spy_mean_rank(
    means: np.ndarray,
    spy_index: int = 0,
) -> np.ndarray:
    """Order regimes by descending SPY-component mean.

    Used as the canonical-fold tiebreaker (and as a fallback when no reference
    is available). Returns a permutation `P` such that `means[P[0], spy_index]`
    is the highest, `means[P[K-1], spy_index]` the lowest.
    """
    return np.argsort(-means[:, spy_index])


def label_agreement(perm_a: np.ndarray, perm_b: np.ndarray) -> float:
    """Fraction of indices where two label sequences agree."""
    a = np.asarray(perm_a)
    b = np.asarray(perm_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.size == 0:
        return 1.0
    return float(np.mean(a == b))
