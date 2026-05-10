"""Stateful streaming forward filter for online HMM inference.

Maintains the most-recent log-α vector so each new observation is processed in
O(K²) instead of O(T·K²). The streaming filter wraps the same forward-filter
recursion as `regime/models/_hmm_core.py` but exposes a `step(obs)` API.

State serialization is pure-numpy (no JAX traces); a Postgres adapter can pickle
or JSON-encode the dict returned by `state_dict`.

The JAX-jitted hot-path version mentioned in PLAN.md Week 9 is a deploy-phase
speedup; correctness lives here in pure-numpy with full test coverage. When we
deploy and need <50ms p99 inference, we re-implement `step` as a `jax.jit`-ed
`scan` with the same input/output shapes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class StreamingFilter:
    """Stateful forward filter (online HMM posterior).

    Args:
        log_pi: log initial-state distribution, shape (K,).
        log_A:  log transition matrix, shape (K, K). A[i, j] = P(s_{t+1}=j | s_t=i).
    """

    def __init__(self, log_pi: np.ndarray, log_A: np.ndarray) -> None:
        log_pi = np.asarray(log_pi, dtype=np.float64).reshape(-1)
        log_A = np.asarray(log_A, dtype=np.float64)
        if log_A.shape != (len(log_pi), len(log_pi)):
            raise ValueError(f"log_A shape {log_A.shape} != ({len(log_pi)}, {len(log_pi)})")
        self.log_pi = log_pi
        self.log_A = log_A
        self.K = len(log_pi)
        self.log_alpha: np.ndarray | None = None
        self.t: int = 0

    def step(self, log_emission: np.ndarray) -> np.ndarray:
        """Process one new observation given its (already-computed) log-emissions.

        `log_emission` has shape (K,) — log p(y_t | s_t = k) for each state.
        Returns P(s_t | y_{1:t}), shape (K,).
        """
        log_e = np.asarray(log_emission, dtype=np.float64).reshape(-1)
        if log_e.shape != (self.K,):
            raise ValueError(f"log_emission shape {log_e.shape} != ({self.K},)")

        if self.log_alpha is None:
            unnorm = self.log_pi + log_e
        else:
            log_pred = np.asarray(logsumexp(self.log_alpha[:, None] + self.log_A, axis=0))
            unnorm = log_pred + log_e

        norm = float(np.asarray(logsumexp(unnorm)))
        log_alpha_new = unnorm - norm
        self.log_alpha = log_alpha_new
        self.t += 1
        return np.exp(log_alpha_new)

    def reset(self) -> None:
        """Restart the filter to the prior."""
        self.log_alpha = None
        self.t = 0

    def state_dict(self) -> dict:
        return {
            "log_pi": self.log_pi.tolist(),
            "log_A": self.log_A.tolist(),
            "log_alpha": self.log_alpha.tolist() if self.log_alpha is not None else None,
            "t": int(self.t),
        }

    def load_state_dict(self, state: dict) -> None:
        self.log_pi = np.asarray(state["log_pi"], dtype=np.float64)
        self.log_A = np.asarray(state["log_A"], dtype=np.float64)
        self.K = len(self.log_pi)
        la = state.get("log_alpha")
        self.log_alpha = np.asarray(la, dtype=np.float64) if la is not None else None
        self.t = int(state.get("t", 0))
