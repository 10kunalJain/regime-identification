"""Streaming-filter tests: equivalence with batch forward, restart correctness."""

from __future__ import annotations

import numpy as np
import pytest

from regime.models._hmm_core import (
    forward_filter,
    gaussian_log_emissions,
    sample_gaussian_hmm,
)
from regime.runtime.filter import StreamingFilter


def _toy_hmm(seed: int = 0, T: int = 200):
    rng = np.random.default_rng(seed)
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
    )
    means = np.array([[0.0], [1.0], [-1.0]])
    covs = np.tile(np.eye(1), (3, 1, 1)) * 0.5
    _, X = sample_gaussian_hmm(pi, A, means, covs, T, rng)
    log_emissions = gaussian_log_emissions(X, means, covs)
    return pi, A, log_emissions


def test_streaming_matches_batch_forward():
    pi, A, log_emissions = _toy_hmm(seed=1, T=200)
    log_pi = np.log(pi)
    log_A = np.log(A)

    batch = forward_filter(log_emissions, log_pi, log_A)

    sf = StreamingFilter(log_pi, log_A)
    streamed = np.stack([sf.step(log_emissions[t]) for t in range(len(log_emissions))], axis=0)

    np.testing.assert_allclose(streamed, batch, atol=1e-12)


def test_streaming_step_normalizes():
    pi, A, log_emissions = _toy_hmm(seed=2, T=50)
    sf = StreamingFilter(np.log(pi), np.log(A))
    for t in range(len(log_emissions)):
        post = sf.step(log_emissions[t])
        assert post.sum() == pytest.approx(1.0, abs=1e-12)


def test_streaming_state_dict_roundtrip():
    pi, A, log_emissions = _toy_hmm(seed=3, T=80)
    sf1 = StreamingFilter(np.log(pi), np.log(A))
    for t in range(50):
        sf1.step(log_emissions[t])

    state = sf1.state_dict()
    sf2 = StreamingFilter(np.log(pi), np.log(A))
    sf2.load_state_dict(state)

    # Continue both filters with the remaining observations.
    for t in range(50, len(log_emissions)):
        p1 = sf1.step(log_emissions[t])
        p2 = sf2.step(log_emissions[t])
        np.testing.assert_allclose(p1, p2, atol=1e-12)
    assert sf1.t == sf2.t


def test_streaming_reset_returns_to_prior():
    pi, A, log_emissions = _toy_hmm(seed=4, T=20)
    sf = StreamingFilter(np.log(pi), np.log(A))
    for t in range(10):
        sf.step(log_emissions[t])
    sf.reset()
    assert sf.log_alpha is None
    assert sf.t == 0
    p_first = sf.step(log_emissions[0])
    np.testing.assert_allclose(p_first.sum(), 1.0)


def test_streaming_validates_log_emission_shape():
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    sf = StreamingFilter(np.log(pi), np.log(A))
    with pytest.raises(ValueError, match="shape"):
        sf.step(np.zeros(3))


def test_streaming_validates_log_a_shape():
    with pytest.raises(ValueError, match="shape"):
        StreamingFilter(np.array([0.5, 0.5]), np.array([[0.9, 0.1]]))
