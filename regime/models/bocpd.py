"""Bayesian Online Change-Point Detection (Adams & MacKay 2007).

Univariate observations with conjugate Normal–Inverse-Gamma prior; closed-form
Student-t predictive density. At every timestep the algorithm maintains a
posterior over run length P(r_t | y_{1:t}); at change points the run-length
distribution collapses toward zero, and the change probability `P(r_t = 0)`
spikes.

Implements `ChangePointModel`. The native features per timestep are:
  - change_prob: posterior probability that t is a change point.
  - expected_run_length: E[r_t | y_{1:t}].
  - run_length_entropy: H[r_t | y_{1:t}].

These are *not* P(state). The ensemble crisis-head consumes them as auxiliary
features alongside state-based methods' P(crisis), per Q7's two-protocol design.

Implementation note: pure numpy with O(T²) worst case (run-length list grows
unboundedly). We truncate at `max_run_length` (default 1000) — after which
low-probability run lengths are dropped. For the JAX-jitted hot-path version
referenced in PLAN.md Week 4, we'll convert this implementation later once the
algorithm is working correctly here.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.special import gammaln, logsumexp

_LOG = logging.getLogger(__name__)
_TINY = 1e-300


class Bocpd:
    """Bayesian online change-point detector. Implements `ChangePointModel`."""

    def __init__(
        self,
        feature_columns: tuple[str, ...] = ("ret_1d",),
        hazard_lambda: float = 250.0,
        max_run_length: int = 1000,
        empirical_bayes: bool = True,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 2.0,
        prior_beta: float = 1.0,
    ) -> None:
        if len(feature_columns) != 1:
            raise ValueError("Bocpd is univariate; got multi-column feature_columns")
        self.feature_columns = feature_columns
        self.hazard_lambda = float(hazard_lambda)
        self.max_run_length = int(max_run_length)
        self.empirical_bayes = bool(empirical_bayes)
        self.prior_mu = float(prior_mu)
        self.prior_kappa = float(prior_kappa)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self._params: dict | None = None

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        """Empirical-Bayes prior calibration from training data (or use the
        passed-in priors when ``empirical_bayes=False``)."""
        y = self._extract(features, train_idx)
        if self.empirical_bayes and len(y) > 1:
            mu = float(y.mean())
            var = float(y.var()) + _TINY
            self._params = {
                "mu_0": mu,
                "kappa_0": 1.0,
                "alpha_0": 2.0,
                # Choose beta_0 so the prior expected variance = var
                # Inverse-Gamma(alpha, beta) has mean beta / (alpha - 1).
                "beta_0": var * (2.0 - 1.0),
                "hazard_lambda": self.hazard_lambda,
            }
        else:
            self._params = {
                "mu_0": self.prior_mu,
                "kappa_0": self.prior_kappa,
                "alpha_0": self.prior_alpha,
                "beta_0": self.prior_beta,
                "hazard_lambda": self.hazard_lambda,
            }

    def native_features(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Run BOCPD on the indexed observations; return (T, 3) feature matrix."""
        if self._params is None:
            raise RuntimeError("model not fit")
        y = self._extract(features, idx)
        return _run_bocpd(y, self._params, self.max_run_length)

    def state_dict(self) -> dict:
        return {
            "fitted": self._params is not None,
            "feature_columns": list(self.feature_columns),
            "hazard_lambda": self.hazard_lambda,
            "max_run_length": self.max_run_length,
            "empirical_bayes": self.empirical_bayes,
            "params": self._params,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state.get("fitted"):
            self._params = None
            return
        self.feature_columns = tuple(state["feature_columns"])
        self.hazard_lambda = float(state["hazard_lambda"])
        self.max_run_length = int(state["max_run_length"])
        self.empirical_bayes = bool(state["empirical_bayes"])
        self._params = dict(state["params"])

    def _extract(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        col = self.feature_columns[0]
        mask = np.zeros(features.height, dtype=bool)
        mask[idx] = True
        sub = features.select(col).filter(pl.Series(mask)).drop_nulls()
        return sub.to_numpy().reshape(-1).astype(np.float64)


def _student_t_log_pdf(
    y: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Posterior-predictive Student-t log-pdf for Normal-IG conjugate update.

    For NIG(mu, kappa, alpha, beta), the marginal predictive is Student-t with
    df = 2*alpha, location = mu, scale² = beta * (kappa + 1) / (alpha * kappa).
    """
    df = 2.0 * alpha
    scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
    z = (y - mu) ** 2 / scale_sq
    log_norm = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(df * np.pi * scale_sq)
    return log_norm - 0.5 * (df + 1.0) * np.log1p(z / df)


def _run_bocpd(y: np.ndarray, params: dict, max_run_length: int) -> np.ndarray:
    """Core recursion. Returns shape (T, 3): change_prob, expected_rl, entropy."""
    mu0 = params["mu_0"]
    kappa0 = params["kappa_0"]
    alpha0 = params["alpha_0"]
    beta0 = params["beta_0"]
    hazard = 1.0 / params["hazard_lambda"]

    T = len(y)
    out = np.zeros((T, 3), dtype=np.float64)

    # Active hypothesis state (one entry per run length).
    log_joint = np.array([0.0])  # log P(r_0 = 0) = 0
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    log_hazard = np.log(hazard)
    log_one_minus_hazard = np.log1p(-hazard)

    for t in range(T):
        y_t = float(y[t])

        # Predictive log-pdf for each active run.
        log_pred = _student_t_log_pdf(y_t, mu, kappa, alpha, beta)

        # Changepoint posterior weight (mass moved to r=0).
        log_change = float(np.asarray(logsumexp(log_joint + log_pred + log_hazard)))

        # Growth posterior weights.
        log_growth = log_joint + log_pred + log_one_minus_hazard

        # New joint distribution: r=0 (changepoint) prepended to growth.
        new_log_joint = np.concatenate(([log_change], log_growth))

        # Suff-stat update: each old run k becomes new run k+1 with y_t absorbed.
        n_old = len(mu)
        new_kappa = np.concatenate(([kappa0], kappa + 1.0))
        new_mu = np.concatenate(([mu0], (kappa * mu + y_t) / (kappa + 1.0)))
        new_alpha = np.concatenate(([alpha0], alpha + 0.5))
        diff = y_t - mu
        new_beta = np.concatenate(([beta0], beta + 0.5 * kappa / (kappa + 1.0) * diff**2))

        # Truncate to max_run_length (keep most-probable hypotheses).
        if len(new_log_joint) > max_run_length:
            keep = np.argsort(new_log_joint)[-max_run_length:]
            keep.sort()  # preserve insertion order for interpretability
            new_log_joint = new_log_joint[keep]
            new_mu = new_mu[keep]
            new_kappa = new_kappa[keep]
            new_alpha = new_alpha[keep]
            new_beta = new_beta[keep]

        log_joint = new_log_joint
        mu = new_mu
        kappa = new_kappa
        alpha = new_alpha
        beta = new_beta

        # Posterior over run length and feature outputs.
        log_p_y = float(np.asarray(logsumexp(log_joint)))
        post = np.exp(log_joint - log_p_y)
        out[t, 0] = post[0]
        r_values = np.arange(len(post), dtype=np.float64)
        out[t, 1] = float((r_values * post).sum())
        out[t, 2] = float(-(post * np.log(np.maximum(post, _TINY))).sum())

        # Suppress underflow from accumulated joint
        if log_p_y < -700.0:
            log_joint = log_joint - log_p_y

        n_old += 0  # silence linter's unused-variable warning

    return out
