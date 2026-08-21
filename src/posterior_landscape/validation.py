"""Fast, read-only validation of an input/configuration pair."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import Settings
from .io import open_density_store, validate_store


def validate_configuration(settings: Settings) -> dict[str, Any]:
    """Return a compact preflight report without running feature extraction."""

    if settings.analysis.run == "1d" and not settings.one_dimensional.enabled:
        raise ValueError(
            "No analysis is enabled: analysis.run=1d conflicts with "
            "one_dimensional.enabled=false."
        )

    with open_density_store(
        settings.input.file,
        input_settings=settings.input,
        association_settings=settings.association,
    ) as store:
        grid = validate_store(
            store,
            log_base=settings.analysis.log_base,
            geometry=settings.analysis.geometry,
            feature_measure=settings.analysis.feature_measure,
            input_settings=settings.input,
        )
        if settings.analysis.ordered_tails and not grid.ordered_domain:
            raise ValueError(
                "analysis.ordered_tails=true requires coordinate2 < coordinate1 "
                "throughout the valid domain."
            )
        if settings.one_dimensional.enabled and (
            np.any(grid.m1 <= 0.0) or np.any(grid.m2 <= 0.0)
        ):
            raise ValueError(
                "The independent 1D finder requires positive coordinates."
            )
        first_weights = np.asarray(grid.input_weight1, dtype=float)
        second_weights = np.asarray(grid.input_weight2, dtype=float)
        cell_weights = np.multiply.outer(first_weights, second_weights) * grid.mask
        indices = np.unique(
            np.linspace(0, store.number_draws - 1, min(16, store.number_draws)).astype(int)
        )
        integrals = np.asarray(
            [
                np.sum(
                    np.where(grid.mask, np.asarray(store.draws[int(index)]), 0.0)
                    * cell_weights
                )
                for index in indices
            ],
            dtype=float,
        )
        if settings.analysis.detect_shoulders and (
            np.any(grid.m1 <= 0.0) or np.any(grid.m2 <= 0.0)
        ):
            raise ValueError(
                "analysis.detect_shoulders=true requires positive coordinates."
            )
        warnings: list[str] = []
        if settings.input.domain == "legacy":
            warnings.append(
                "input.domain=legacy preserves the v0.7 mask-or-ordered convention; "
                "new inputs should choose full, ordered, or mask explicitly."
            )
        return {
            "input_file": str(settings.input.file),
            "draws": store.number_draws,
            "shape": list(grid.shape),
            "coordinate1": {
                "name": grid.coordinate1_name,
                "range": [float(grid.m1[0]), float(grid.m1[-1])],
            },
            "coordinate2": {
                "name": grid.coordinate2_name,
                "range": [float(grid.m2[0]), float(grid.m2[-1])],
            },
            "domain": settings.input.domain,
            "valid_fraction": float(np.mean(grid.mask)),
            "ordered_domain": grid.ordered_domain,
            "input_density_measure": grid.input_density_measure,
            "feature_measure": grid.feature_measure,
            "geometry": grid.geometry,
            "sampled_normalization": {
                "minimum": float(np.min(integrals)),
                "maximum": float(np.max(integrals)),
                "maximum_absolute_error_from_one": float(
                    np.max(np.abs(integrals - 1.0))
                ),
            },
            "external_parameter": {
                "dataset": settings.association.dataset,
                "name": settings.association.parameter_name,
                "available": store.h0 is not None,
                "chains": (
                    int(np.unique(store.chain_id).size)
                    if store.chain_id is not None
                    else 0
                ),
            },
            "one_dimensional_enabled": settings.one_dimensional.enabled,
            "shoulders_enabled": settings.analysis.detect_shoulders,
            "ordered_tails_enabled": settings.analysis.ordered_tails,
            "output_profile": settings.output.profile,
            "warnings": warnings,
        }
