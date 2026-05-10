"""Feature definitions.

Each per-ticker module registers its features at import time via
`regime.features.registry.register`. Importing this package transitively imports
every definition module so the registry is fully populated.

Cross-sectional, VIX, macro, and FF feature builders live in this package too
but are exposed via dedicated functions (build_*_features) rather than the
per-ticker registry, since they operate on multi-ticker / multi-series inputs.
"""

from regime.features.definitions import (  # noqa: F401
    dispersion,
    factors,
    macro,
    returns,
    trend,
    vix,
    volatility,
)
