"""Fast, read-only validation of an input/configuration pair."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import Settings
from .io import input_to_feature_factor, open_density_store, validate_store


def validate_configuration(settings: Settings, *, all_draws: bool = False) -> dict[str, Any]:
    """Audit selected quadrature without extraction; optionally stream every draw."""

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
                "analysis.ordered_tails=true requires coordinate2 <= coordinate1 "
                "throughout the valid domain."
            )
        if settings.one_dimensional.enabled and (
            np.any(grid.m1 <= 0.0) or np.any(grid.m2 <= 0.0)
        ):
            raise ValueError(
                "The independent 1D finder requires positive coordinates."
            )
        cell_weights = grid.input_cell_weights
        indices = (
            np.arange(store.number_draws)
            if all_draws
            else np.unique(np.linspace(
                0, store.number_draws - 1, min(16, store.number_draws)
            ).astype(int))
        )
        integrals = np.empty(indices.size, dtype=float)
        conversion_integrals = np.empty(indices.size, dtype=float)
        maximum_post_error = 0.0
        maximum_marginal_error = 0.0
        maximum_raw_marginal_error = 0.0
        maximum_marginal_conservation_error = 0.0
        factor = input_to_feature_factor(grid)
        # Small batches avoid materializing the full PixelPop posterior.
        for start in range(0, indices.size, 64):
            chosen = indices[start:start + 64]
            values = np.stack([
                np.asarray(store.draws[int(index)], dtype=float) for index in chosen
            ])
            valid = values[:, grid.mask]
            if np.any(~np.isfinite(valid)) or np.any(valid < 0.0):
                raise ValueError("Audited density draws contain invalid active values.")
            values = np.where(grid.mask[None, :, :], values, 0.0)
            raw = np.einsum("bij,ij->b", values, cell_weights)
            if np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
                raise ValueError("Audited density draws have invalid normalization.")
            integrals[start:start + chosen.size] = raw
            first = np.einsum("bij,j->bi", values, grid.input_weight2)
            second = np.einsum("bij,i->bj", values, grid.input_weight1)
            marginal_integrals = (
                np.einsum("bi,i->b", first, grid.input_weight1),
                np.einsum("bj,j->b", second, grid.input_weight2),
            )
            maximum_raw_marginal_error = max(maximum_raw_marginal_error, *(
                float(np.max(np.abs(result - 1.0))) for result in marginal_integrals
            ))
            maximum_marginal_conservation_error = max(
                maximum_marginal_conservation_error, *(
                    float(np.max(np.abs(result - raw))) for result in marginal_integrals
                ),
            )
            values /= raw[:, None, None]
            values *= factor[None, :, :]
            converted = np.einsum("bij,ij->b", values, grid.cell_weights)
            if np.any(~np.isfinite(converted)) or np.any(converted <= 0.0):
                raise ValueError("Audited conversion has invalid normalization.")
            conversion_integrals[start:start + chosen.size] = converted
            values /= converted[:, None, None]
            maximum_post_error = max(maximum_post_error, float(np.max(np.abs(
                np.einsum("bij,ij->b", values, grid.cell_weights) - 1.0
            ))))
            first = np.einsum("bij,j->bi", values, grid.weight2)
            second = np.einsum("bij,i->bj", values, grid.weight1)
            maximum_marginal_error = max(maximum_marginal_error, *(
                float(np.max(np.abs(result - 1.0))) for result in (
                    np.einsum("bi,i->b", first, grid.weight1),
                    np.einsum("bj,j->b", second, grid.weight2),
                )
            ))
        quantiles = np.quantile(integrals, (0.0, 0.05, 0.5, 0.95, 1.0))
        normalization_report = {
            "minimum": float(quantiles[0]),
            "percentile_5": float(quantiles[1]),
            "median": float(quantiles[2]),
            "percentile_95": float(quantiles[3]),
            "maximum": float(quantiles[4]),
            "maximum_absolute_error_from_one": float(np.max(np.abs(integrals - 1.0))),
        }
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
            "grid_location": grid.grid_location,
            "grid_measure": grid.grid_measure,
            "quadrature": grid.quadrature_metadata,
            "audited_draws": int(indices.size),
            "audit_scope": "all_draws" if all_draws else "representative_draws",
            "active_mask_cells": grid.quadrature_metadata["active_mask_cells"],
            "active_diagonal_cells": grid.quadrature_metadata["active_diagonal_cells"],
            "sampled_normalization": normalization_report,
            "raw_normalization": normalization_report,
            "maximum_input_normalization_error": normalization_report[
                "maximum_absolute_error_from_one"
            ],
            "maximum_conversion_normalization_error": float(np.max(np.abs(
                conversion_integrals - 1.0
            ))),
            "maximum_post_normalization_error": maximum_post_error,
            "maximum_marginal_normalization_error": maximum_marginal_error,
            "maximum_raw_marginal_normalization_error": maximum_raw_marginal_error,
            "maximum_marginal_conservation_error": maximum_marginal_conservation_error,
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
