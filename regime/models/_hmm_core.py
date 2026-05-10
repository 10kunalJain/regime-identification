"""Pure-numpy HMM primitives: Gaussian emissions, forward-only filter, forward-backward.

Used by `HmmGaussian` to compute filtered (P(s_t | y_{1:t})) and smoothed
(P(s_t | y_{1:T})) posteriors from fitted parameters. We don't depend on
hmmlearn for these because (a) hmmlearn's `predict_proba` is smoothed only —
no online filtered output — and (b) implementing the recursions ourselves
gives us exact control over the filtered output, which is the headline
metric in the fair-evaluation protocol.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

LOG_2_PI = float(np.log(2.0 * np.pi))


def gaussian_log_emissions(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Log P(y_t | s_t = k) for every (t, k).

    Args:
        X: shape (T, D) observations.
        means: shape (K, D).
        covs: shape (K, D, D), full covariances.

    Returns:
        shape (T, K) log emission likelihoods.
    """
    T, D = X.shape
    K = means.shape[0]
    log_emit = np.empty((T, K), dtype=np.float64)
    for k in range(K):
        diff = X - means[k]
        cov = covs[k]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # Degenerate covariance — fall back to a heavy penalty rather than NaN.
            log_emit[:, k] = -np.inf
            continue
        cov_inv = np.linalg.inv(cov)
        quad = np.einsum("td,de,te->t", diff, cov_inv, diff)
        log_emit[:, k] = -0.5 * (D * LOG_2_PI + logdet + quad)
    return log_emit


def forward_filter(log_emissions: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    """Filtered posterior P(s_t | y_{1:t}) for each t.

    Args:
        log_emissions: shape (T, K).
        log_pi: shape (K,) log initial-state distribution.
        log_A: shape (K, K) log transition matrix; A[i, j] = P(s_{t+1}=j | s_t=i).

    Returns:
        shape (T, K) filtered posterior.
    """
    T, K = log_emissions.shape
    log_alpha = np.empty((T, K), dtype=np.float64)

    log_alpha[0] = log_pi + log_emissions[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, T):
        # log P(s_t = j | y_{1:t-1}) = logsumexp_i (log_alpha[t-1, i] + log_A[i, j])
        log_pred = logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0)
        unnorm = log_pred + log_emissions[t]
        log_alpha[t] = unnorm - logsumexp(unnorm)

    return np.exp(log_alpha)


def forward_backward(
    log_emissions: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
) -> np.ndarray:
    """Smoothed posterior P(s_t | y_{1:T}) for each t (Baum-Welch E-step).

    Returns shape (T, K).
    """
    T, K = log_emissions.shape

    # Forward pass (un-normalized to retain joint likelihood)
    log_alpha = np.empty((T, K), dtype=np.float64)
    log_alpha[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0) + log_emissions[t]

    # Backward pass
    log_beta = np.zeros((T, K), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A + log_emissions[t + 1, None, :] + log_beta[t + 1, None, :], axis=1
        )

    log_gamma = log_alpha + log_beta
    norm = np.asarray(logsumexp(log_gamma, axis=1, keepdims=True))
    log_gamma -= norm
    return np.exp(log_gamma)


def student_t_log_emissions_univariate(
    y: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, nu: float
) -> np.ndarray:
    """Univariate Student-t log emission density for K states.

    Args:
        y: shape (T,) observations.
        mus: shape (K,) per-state location.
        sigmas: shape (K,) per-state scale (>0).
        nu: degrees-of-freedom (shared across states, fixed).

    Returns:
        shape (T, K) log p(y_t | s_t = k).
    """
    from scipy.special import gammaln

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mus = np.asarray(mus, dtype=np.float64).reshape(-1)
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)

    coef = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
    diff = (y[:, None] - mus[None, :]) / sigmas[None, :]
    log_kern = -0.5 * (nu + 1.0) * np.log1p(diff**2 / nu)
    log_norm = coef - np.log(sigmas)
    return log_kern + log_norm[None, :]


def forward_backward_xi(
    log_emissions: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward with pairwise posteriors for HMM EM M-step.

    Returns:
        gamma: shape (T, K) — P(s_t = k | y).
        xi:    shape (T-1, K, K) — P(s_{t-1}=i, s_t=j | y).
        log_lik: scalar — log P(y).
    """
    T, K = log_emissions.shape

    log_alpha = np.empty((T, K), dtype=np.float64)
    log_alpha[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0) + log_emissions[t]

    log_beta = np.zeros((T, K), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A + log_emissions[t + 1, None, :] + log_beta[t + 1, None, :], axis=1
        )

    log_lik = float(np.asarray(logsumexp(log_alpha[-1])))

    log_gamma = log_alpha + log_beta - log_lik
    gamma = np.exp(log_gamma)

    # xi[t, i, j] = P(s_t = i, s_{t+1} = j | y), for t = 0..T-2
    log_xi = (
        log_alpha[:-1, :, None]
        + log_A[None, :, :]
        + log_emissions[1:, None, :]
        + log_beta[1:, None, :]
        - log_lik
    )
    xi = np.exp(log_xi)
    return gamma, xi, log_lik


def sample_gaussian_hmm(
    pi: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    T: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a state sequence and observation sequence from a Gaussian HMM.

    Returns (states shape (T,), observations shape (T, D)).
    """
    K, D = means.shape
    states = np.empty(T, dtype=np.int64)
    obs = np.empty((T, D), dtype=np.float64)
    states[0] = rng.choice(K, p=pi)
    obs[0] = rng.multivariate_normal(means[states[0]], covs[states[0]])
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
        obs[t] = rng.multivariate_normal(means[states[t]], covs[states[t]])
    return states, obs
