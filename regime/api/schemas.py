"""Pydantic schemas for the API.

Per ARCHITECTURE.md §5 — response models for `/regime/now`, `/regime/path`, and
`/forecast`. Field naming matches the spec exactly so a frontend can consume
the same shape regardless of the backing state store.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class RegimePosterior(BaseModel):
    """Single timestep's regime posterior + calibrated crisis probability."""

    model_config = ConfigDict(frozen=True)

    as_of: date = Field(..., description="Date this posterior corresponds to.")
    regime_probs_uncal: dict[str, float] = Field(
        ..., description="Uncalibrated three-state posterior, e.g. {'calm_bull': 0.83, ...}."
    )
    crisis_prob_21d_cal: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated P(crisis within 21 days)."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Entropy-derived confidence in [0, 1]."
    )
    method: str = Field(..., description="Identifier of the model that produced this posterior.")
    version: str = Field(..., description="Model version hash.")


class ForecastDistribution(BaseModel):
    """Multi-horizon regime forecast."""

    model_config = ConfigDict(frozen=True)

    as_of: date
    horizon_days: int = Field(..., ge=1, le=63)
    expected_regime_probs: dict[str, float]
    expected_log_return: float
    method: str
    version: str
