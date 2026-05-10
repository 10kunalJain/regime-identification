"""Feature engineering layer."""

# Importing definitions triggers registration in regime.features.registry.REGISTRY.
# Keep this import at module level so any consumer of `regime.features` sees a
# fully-populated registry.
from regime.features import definitions  # noqa: F401
from regime.features.registry import REGISTRY, Feature, all_features, register

__all__ = ["REGISTRY", "Feature", "all_features", "register"]
