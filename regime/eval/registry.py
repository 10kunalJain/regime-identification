"""Model registry for the Day-2 cross-method benchmark.

Six methods are wired in, split across the two protocols (CLAUDE.md §Stack;
ARCHITECTURE.md §4):

  - `state` (StateRegimeModel): HmmGaussian, MsarT, SparseJumpModel, JointHmm.
    Output P(state); the runner picks the crisis state per fold by training-set
    correlation with the forward-drawdown label (with a low-mean-return
    fallback for early folds that contain no positive labels).
  - `changepoint` (ChangePointModel): Bocpd, WassersteinKmeans.
    Output native features; the runner fits a thin per-fold logistic head on
    those features to produce a comparable per-method P(crisis) headline
    scalar — exactly what CLAUDE.md's two-protocol design calls for, and what
    the Day-3 ensemble crisis head will consume jointly across all methods.

Pre-registration. The hyperparameters below are the initial-commit defaults
of each model class (locked in commit `bf3f771`) except for two deliberate
overrides documented inline: the univariate observation column (`ret_SPY` —
the daily log total return of SPY in the wide dataframe built by
`regime.data.joint_dataset.build_wide_dataframe`) and the multivariate
realized-vol companion column (`rv_21d_SPY` — 21-day rolling std of SPY log
returns, computed by the benchmark script before the harness is invoked).
These choices put all univariate baselines on the same observation and avoid
forced K=3 wrappers on the change-point methods. Any subsequent edit must be
explicit and noted in the commit message; no post-hoc tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from regime.models.base import ChangePointModel, StateRegimeModel
from regime.models.bocpd import Bocpd
from regime.models.hmm_gaussian import HmmGaussian
from regime.models.joint_hmm import JointHmm
from regime.models.msar_t import MsarT
from regime.models.sparse_jump import SparseJumpModel
from regime.models.wasserstein_kmeans import WassersteinKmeans

# Univariate observation = SPY log return; level playing field across the
# univariate baselines. The wide dataframe from `build_wide_dataframe` already
# carries `ret_SPY`; `rv_21d_SPY` is derived once by the benchmark script.
UNIVARIATE_RETURN_COLUMN = "ret_SPY"
UNIVARIATE_RV_COLUMN = "rv_21d_SPY"

MethodKind = Literal["state", "changepoint"]
ModelFactory = Callable[[tuple[str, ...], tuple[str, ...]], StateRegimeModel | ChangePointModel]


@dataclass(frozen=True)
class RegisteredMethod:
    """A benchmark-registered method with its factory and output schema.

    `factory(obs_cols, factor_cols)` returns a fresh, unfit model. The factor
    columns are only used by the joint HMM (multivariate observation + FF
    factor regressors); univariate baselines ignore them.
    """

    name: str
    kind: MethodKind
    factory: ModelFactory
    raw_feature_names: tuple[str, ...]


def _hmm_gaussian_factory(_obs: tuple[str, ...], _factors: tuple[str, ...]) -> HmmGaussian:
    return HmmGaussian(
        K=3,
        feature_columns=(UNIVARIATE_RETURN_COLUMN, UNIVARIATE_RV_COLUMN),
        n_restarts=5,
        max_iter=200,
        tol=1e-4,
        random_state=42,
    )


def _msar_t_factory(_obs: tuple[str, ...], _factors: tuple[str, ...]) -> MsarT:
    return MsarT(
        K=3,
        feature_columns=(UNIVARIATE_RETURN_COLUMN,),
        nu=5.0,
        n_restarts=5,
        max_iter=100,
        tol=1e-5,
        random_state=42,
    )


def _sparse_jump_factory(_obs: tuple[str, ...], _factors: tuple[str, ...]) -> SparseJumpModel:
    return SparseJumpModel(
        K=3,
        feature_columns=(UNIVARIATE_RETURN_COLUMN, UNIVARIATE_RV_COLUMN),
        jump_penalty=0.5,
        n_iter=50,
        n_restarts=5,
        random_state=42,
    )


def _joint_hmm_factory(obs: tuple[str, ...], factors: tuple[str, ...]) -> JointHmm:
    return JointHmm(
        K=3,
        observation_columns=obs,
        factor_columns=factors,
        latent_factor_rank=3,
        n_restarts=3,
        max_iter=50,
        random_state=42,
    )


def _bocpd_factory(_obs: tuple[str, ...], _factors: tuple[str, ...]) -> Bocpd:
    return Bocpd(
        feature_columns=(UNIVARIATE_RETURN_COLUMN,),
        hazard_lambda=250.0,
        max_run_length=1000,
        empirical_bayes=True,
    )


def _wasserstein_kmeans_factory(
    _obs: tuple[str, ...], _factors: tuple[str, ...]
) -> WassersteinKmeans:
    # n_iter=10 and n_restarts=2 are *downgrades* from the model class defaults
    # (20 / 3), chosen for computational tractability: the medoid update is
    # O(m²) per cluster per iteration, and on the largest expanding-window fold
    # (~5800 train rows) the default would push wall-time past 1 h for this
    # method alone. With n_iter=10 / n_restarts=2 the sweep completes in
    # ~15 min and the medoid assignments are stable at this level (kmeans-like
    # iterations typically converge in <10 steps). No results-driven tuning —
    # downgrade-for-tractability is documented here for the audit trail.
    return WassersteinKmeans(
        K=3,
        feature_columns=(UNIVARIATE_RETURN_COLUMN,),
        window=21,
        n_iter=10,
        n_restarts=2,
        n_projections=50,
        random_state=42,
    )


REGISTRY: tuple[RegisteredMethod, ...] = (
    RegisteredMethod(
        name="hmm_gaussian",
        kind="state",
        factory=_hmm_gaussian_factory,
        raw_feature_names=("filtered_0", "filtered_1", "filtered_2"),
    ),
    RegisteredMethod(
        name="msar_t",
        kind="state",
        factory=_msar_t_factory,
        raw_feature_names=("filtered_0", "filtered_1", "filtered_2"),
    ),
    RegisteredMethod(
        name="sparse_jump",
        kind="state",
        factory=_sparse_jump_factory,
        raw_feature_names=("filtered_0", "filtered_1", "filtered_2"),
    ),
    RegisteredMethod(
        name="joint_hmm",
        kind="state",
        factory=_joint_hmm_factory,
        raw_feature_names=("filtered_0", "filtered_1", "filtered_2"),
    ),
    RegisteredMethod(
        name="bocpd",
        kind="changepoint",
        factory=_bocpd_factory,
        raw_feature_names=("change_prob", "expected_run_length", "run_length_entropy"),
    ),
    RegisteredMethod(
        name="wasserstein_kmeans",
        kind="changepoint",
        factory=_wasserstein_kmeans_factory,
        raw_feature_names=("dist_0", "dist_1", "dist_2"),
    ),
)


def get(name: str) -> RegisteredMethod:
    for m in REGISTRY:
        if m.name == name:
            return m
    known = ", ".join(m.name for m in REGISTRY)
    raise KeyError(f"method {name!r} not registered; known: {known}")


def all_methods() -> tuple[RegisteredMethod, ...]:
    return REGISTRY


__all__ = [
    "REGISTRY",
    "UNIVARIATE_RETURN_COLUMN",
    "UNIVARIATE_RV_COLUMN",
    "MethodKind",
    "ModelFactory",
    "RegisteredMethod",
    "all_methods",
    "get",
]
