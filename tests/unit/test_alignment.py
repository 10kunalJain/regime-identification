"""Tests for Hungarian + W₂ regime-label alignment."""

from __future__ import annotations

import numpy as np
import pytest

from regime.models.alignment import (
    align_by_spy_mean_rank,
    hungarian_align,
    label_agreement,
    w2_squared_gaussian,
)


def test_w2_zero_for_identical_gaussians():
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    assert w2_squared_gaussian(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-10)


def test_w2_translation_only():
    """W₂² between two equal-variance Gaussians equals the squared mean distance."""
    mu_a = np.array([0.0, 0.0])
    mu_b = np.array([3.0, 4.0])
    cov = np.eye(2)
    expected = 25.0  # 3² + 4²
    assert w2_squared_gaussian(mu_a, cov, mu_b, cov) == pytest.approx(expected, rel=1e-6)


def test_hungarian_recovers_permutation():
    """For permuted candidate b = means[perm], align should return inverse(perm).

    Definition: result[k] is the candidate index that aligns with reference k.
    Since means_b[j] == means[perm[j]], we have means_b[j] == means[k] iff
    j == inverse(perm)[k]. So result == inverse(perm).
    """
    means = np.array([[0.0], [1.0], [-1.0]])
    covs = np.array([[[0.1]], [[0.2]], [[0.3]]])

    perm = np.array([2, 0, 1])
    inverse_perm = np.argsort(perm)  # [1, 2, 0]
    means_b = means[perm]
    covs_b = covs[perm]

    result = hungarian_align(means, covs, means_b, covs_b)
    np.testing.assert_array_equal(result, inverse_perm)


def test_hungarian_handles_close_means():
    """When two regimes are close, the alignment still produces a valid permutation."""
    means = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    covs = np.tile(np.eye(2), (3, 1, 1)) * 0.5

    means_b = means[[1, 0, 2]]  # swap first two
    covs_b = covs[[1, 0, 2]]

    result = hungarian_align(means, covs, means_b, covs_b)
    # Result must be a valid permutation regardless of which way it resolves the tie.
    assert sorted(result.tolist()) == [0, 1, 2]


def test_hungarian_size_mismatch_raises():
    means_a = np.zeros((3, 2))
    means_b = np.zeros((4, 2))
    covs_a = np.tile(np.eye(2), (3, 1, 1))
    covs_b = np.tile(np.eye(2), (4, 1, 1))
    with pytest.raises(ValueError):
        hungarian_align(means_a, covs_a, means_b, covs_b)


def test_align_by_spy_mean_rank_descending():
    means = np.array([[0.001, 0.0], [-0.001, 0.0], [0.0005, 0.0]])
    perm = align_by_spy_mean_rank(means, spy_index=0)
    assert perm[0] == 0  # highest SPY mean
    assert perm[-1] == 1  # lowest SPY mean


def test_label_agreement_perfect():
    a = np.array([0, 1, 2, 0, 1])
    assert label_agreement(a, a) == pytest.approx(1.0)


def test_label_agreement_partial():
    a = np.array([0, 1, 2, 0])
    b = np.array([0, 1, 1, 0])
    assert label_agreement(a, b) == pytest.approx(0.75)


def test_label_agreement_shape_mismatch_raises():
    with pytest.raises(ValueError):
        label_agreement(np.array([0, 1]), np.array([0, 1, 2]))
