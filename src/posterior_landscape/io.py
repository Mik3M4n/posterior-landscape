"""Input, validation, quadrature, and compact HDF5 output helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from .config import AssociationSettings, InputSettings

try:  # Imported lazily enough that NPZ-only utilities remain usable.
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal environments.
    h5py = None


def logarithm(values: np.ndarray, base: float) -> np.ndarray:
    return np.log(np.asarray(values, dtype=float)) / np.log(base)


def quadrature_weights(coordinates: np.ndarray) -> np.ndarray:
    """Trapezoidal nodal weights for a strictly increasing 1D grid."""

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("Each coordinate grid must be one-dimensional with >=2 nodes.")
    differences = np.diff(coordinates)
    if np.any(~np.isfinite(coordinates)) or np.any(differences <= 0.0):
        raise ValueError("Coordinate grids must be finite and strictly increasing.")
    result = np.empty_like(coordinates)
    result[0] = 0.5 * differences[0]
    result[-1] = 0.5 * differences[-1]
    if coordinates.size > 2:
        result[1:-1] = 0.5 * (coordinates[2:] - coordinates[:-2])
    return result


@dataclass(frozen=True)
class Grid:
    m1: np.ndarray
    m2: np.ndarray
    ell1: np.ndarray
    ell2: np.ndarray
    geometry1: np.ndarray
    geometry2: np.ndarray
    mask: np.ndarray
    weight1: np.ndarray
    weight2: np.ndarray
    log_base: float
    geometry: str
    feature_measure: str = "log"
    input_density_measure: str = "log"
    domain: str = "legacy"
    ordered_domain: bool = True
    coordinate1_name: str = "m1"
    coordinate2_name: str = "m2"
    coordinate1_label: str = r"$m_1$"
    coordinate2_label: str = r"$m_2$"
    coordinate1_unit: str = ""
    coordinate2_unit: str = ""
    input_weight1: np.ndarray | None = None
    input_weight2: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def cell_weights(self) -> np.ndarray:
        return np.multiply.outer(self.weight1, self.weight2) * self.mask

    @property
    def typical_spacing(self) -> float:
        return float(
            np.sqrt(
                np.median(np.diff(self.geometry1)) * np.median(np.diff(self.geometry2))
            )
        )

    def integrate(self, density: np.ndarray) -> float:
        values = np.where(self.mask, np.asarray(density, dtype=float), 0.0)
        return float(np.sum(values * self.cell_weights))

    def marginals(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.where(self.mask, np.asarray(density, dtype=float), 0.0)
        first = np.einsum("ij,j->i", values, self.weight2, optimize=True)
        second = np.einsum("ij,i->j", values, self.weight1, optimize=True)
        return first, second


class DensityStore(AbstractContextManager["DensityStore"]):
    """Common lazy interface for NPZ and HDF5 posterior fields."""

    path: Path
    draws: Any
    m1: np.ndarray
    m2: np.ndarray
    mask: np.ndarray
    weights: np.ndarray
    h0: np.ndarray | None
    chain_id: np.ndarray | None
    log_base_attribute: float | None
    density_measure_attribute: str | None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.draws.shape)

    @property
    def number_draws(self) -> int:
        return self.shape[0]

    def batch(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self.draws[start:stop], dtype=float)

    def iter_batches(self, batch_size: int) -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, self.number_draws, batch_size):
            yield start, self.batch(start, min(start + batch_size, self.number_draws))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def _domain_mask(
    first: np.ndarray,
    second: np.ndarray,
    supplied: np.ndarray | None,
    domain: str,
) -> np.ndarray:
    if domain == "full":
        if supplied is not None and not np.all(supplied):
            raise ValueError(
                "input.domain=full conflicts with a mask containing false values."
            )
        return np.ones((first.size, second.size), dtype=bool)
    if domain == "ordered":
        expected = second[None, :] < first[:, None]
        if supplied is not None:
            if np.any(supplied & ~expected):
                raise ValueError(
                    "input.domain=ordered requires every valid mask cell to "
                    "satisfy coordinate2 < coordinate1."
                )
            # A supplied mask may further restrict an ordered domain. This is
            # the same mask-first behavior used by v0.7, while making the
            # ordering assumption explicit and testable.
            return supplied
        return expected
    if domain == "mask":
        if supplied is None:
            raise ValueError("input.domain=mask requires the configured mask dataset.")
        return supplied
    # Exact v0.7 convention. New generic configurations should select a domain
    # explicitly rather than relying on this compatibility mode.
    return supplied if supplied is not None else second[None, :] < first[:, None]


class NpzDensityStore(DensityStore):
    def __init__(
        self,
        path: Path,
        input_settings: InputSettings | None = None,
        association_settings: AssociationSettings | None = None,
    ):
        input_settings = input_settings or InputSettings(file=path)
        association_settings = association_settings or AssociationSettings()
        self.path = path
        self._archive = np.load(path, allow_pickle=False)
        required = {
            input_settings.density_dataset,
            input_settings.coordinate1_dataset,
            input_settings.coordinate2_dataset,
        }
        missing = required - set(self._archive.files)
        if missing:
            self._archive.close()
            raise ValueError(f"NPZ input is missing arrays: {sorted(missing)}")
        self.draws = self._archive[input_settings.density_dataset]
        if self.draws.ndim != 3 or self.draws.shape[0] < 1:
            self._archive.close()
            raise ValueError("p must contain at least one draw with shape (B, N1, N2).")
        self.m1 = np.asarray(
            self._archive[input_settings.coordinate1_dataset], dtype=float
        )
        self.m2 = np.asarray(
            self._archive[input_settings.coordinate2_dataset], dtype=float
        )
        supplied_mask = (
            np.asarray(self._archive[input_settings.mask_dataset], dtype=bool)
            if input_settings.mask_dataset in self._archive.files
            else None
        )
        self.mask = _domain_mask(
            self.m1, self.m2, supplied_mask, input_settings.domain
        )
        if input_settings.weights_dataset in self._archive.files:
            self.weights = np.asarray(
                self._archive[input_settings.weights_dataset], dtype=float
            )
        else:
            self.weights = np.full(self.draws.shape[0], 1.0 / self.draws.shape[0])
        self.h0 = (
            np.asarray(self._archive[association_settings.dataset], dtype=float)
            if association_settings.dataset in self._archive.files
            else None
        )
        self.chain_id = (
            np.asarray(self._archive[association_settings.chain_id_dataset])
            if association_settings.chain_id_dataset in self._archive.files
            else None
        )
        self.log_base_attribute = (
            float(np.asarray(self._archive["log_base"]).item())
            if "log_base" in self._archive.files
            else None
        )
        self.density_measure_attribute = (
            str(np.asarray(self._archive["density_measure"]).item())
            if "density_measure" in self._archive.files
            else None
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._archive.close()


class Hdf5DensityStore(DensityStore):
    def __init__(
        self,
        path: Path,
        input_settings: InputSettings | None = None,
        association_settings: AssociationSettings | None = None,
    ):
        input_settings = input_settings or InputSettings(file=path)
        association_settings = association_settings or AssociationSettings()
        if h5py is None:
            raise RuntimeError(
                "HDF5 input requires h5py. Install the package dependencies first."
            )
        self.path = path
        self._file = h5py.File(path, "r")
        required = {
            input_settings.density_dataset,
            input_settings.coordinate1_dataset,
            input_settings.coordinate2_dataset,
        }
        missing = required - set(self._file.keys())
        if missing:
            self._file.close()
            raise ValueError(f"HDF5 input is missing datasets: {sorted(missing)}")
        self.draws = self._file[input_settings.density_dataset]
        if self.draws.ndim != 3 or self.draws.shape[0] < 1:
            self._file.close()
            raise ValueError("p must contain at least one draw with shape (B, N1, N2).")
        self.m1 = np.asarray(
            self._file[input_settings.coordinate1_dataset], dtype=float
        )
        self.m2 = np.asarray(
            self._file[input_settings.coordinate2_dataset], dtype=float
        )
        supplied_mask = (
            np.asarray(self._file[input_settings.mask_dataset], dtype=bool)
            if input_settings.mask_dataset in self._file
            else None
        )
        self.mask = _domain_mask(
            self.m1, self.m2, supplied_mask, input_settings.domain
        )
        self.weights = (
            np.asarray(self._file[input_settings.weights_dataset], dtype=float)
            if input_settings.weights_dataset in self._file
            else np.full(self.draws.shape[0], 1.0 / self.draws.shape[0])
        )
        self.h0 = (
            np.asarray(self._file[association_settings.dataset], dtype=float)
            if association_settings.dataset in self._file
            else None
        )
        self.chain_id = (
            np.asarray(self._file[association_settings.chain_id_dataset])
            if association_settings.chain_id_dataset in self._file
            else None
        )
        attribute = self._file.attrs.get("log_base")
        self.log_base_attribute = float(attribute) if attribute is not None else None
        density_measure = self._file.attrs.get("density_measure")
        if isinstance(density_measure, bytes):
            density_measure = density_measure.decode("utf-8")
        self.density_measure_attribute = (
            str(density_measure) if density_measure is not None else None
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._file.close()


def open_density_store(
    path: str | Path,
    *,
    input_settings: InputSettings | None = None,
    association_settings: AssociationSettings | None = None,
) -> DensityStore:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input density file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return NpzDensityStore(path, input_settings, association_settings)
    if suffix in {".h5", ".hdf5"}:
        return Hdf5DensityStore(path, input_settings, association_settings)
    raise ValueError("Input must be an .h5, .hdf5, or .npz file.")


def validate_store(
    store: DensityStore,
    *,
    log_base: float,
    geometry: str,
    feature_measure: str = "log",
    input_settings: InputSettings | None = None,
    sample_draws: int = 16,
) -> Grid:
    """Validate structural input and a representative set of field values."""

    input_settings = input_settings or InputSettings(file=store.path)
    input_density_measure = input_settings.density_measure

    if len(store.shape) != 3:
        raise ValueError("p must have shape (draw, m1, m2).")
    if store.shape[1:] != (store.m1.size, store.m2.size):
        raise ValueError("p dimensions do not agree with m1 and m2.")
    if store.mask.shape != store.shape[1:]:
        raise ValueError("mask shape does not agree with p.")
    if not np.any(store.mask):
        raise ValueError("The valid domain mask is empty.")
    logarithmic_coordinates_needed = (
        geometry == "log"
        or feature_measure == "log"
        or input_density_measure == "log"
    )
    if logarithmic_coordinates_needed and (
        np.any(store.m1 <= 0.0) or np.any(store.m2 <= 0.0)
    ):
        raise ValueError(
            "Coordinates must be positive when logarithmic geometry or a "
            "logarithmic density measure is selected."
        )
    if store.weights.shape != (store.number_draws,):
        raise ValueError("Posterior weights must have one entry per draw.")
    if np.any(~np.isfinite(store.weights)) or np.any(store.weights < 0.0):
        raise ValueError("Posterior weights must be finite and nonnegative.")
    total_weight = float(store.weights.sum())
    if total_weight <= 0.0:
        raise ValueError("Posterior weights sum to zero.")
    store.weights[:] = store.weights / total_weight

    if store.h0 is not None:
        if store.h0.shape != (store.number_draws,):
            raise ValueError(
                "The external parameter must have one aligned value per density draw."
            )
        if np.any(~np.isfinite(store.h0)):
            raise ValueError("External-parameter values must be finite.")
        if store.chain_id is None:
            store.chain_id = np.zeros(store.number_draws, dtype=np.int64)
        if store.chain_id.shape != (store.number_draws,):
            raise ValueError("chain_id must have one aligned value per density draw.")
    elif store.chain_id is not None:
        raise ValueError(
            "chain_id was supplied without the aligned external-parameter array."
        )

    if store.log_base_attribute is not None and not np.isclose(
        store.log_base_attribute, log_base
    ):
        raise ValueError(
            "log_base in settings does not match the input file attribute "
            f"({log_base:g} versus {store.log_base_attribute:g})."
        )

    if store.density_measure_attribute is not None:
        declared = store.density_measure_attribute.strip().lower()
        if declared in {"log", "dlogm1_dlogm2", "dlogx1_dlogx2"}:
            declared = "log"
        elif declared in {"linear", "dm1_dm2", "dx1_dx2"}:
            declared = "linear"
        else:
            raise ValueError(
                "Unrecognized density_measure attribute in the input file: "
                f"{store.density_measure_attribute!r}."
            )
        if declared != input_density_measure:
            raise ValueError(
                "input.density_measure does not match the input file attribute "
                f"({input_density_measure!r} versus {declared!r})."
            )

    if feature_measure not in {"log", "linear"}:
        raise ValueError("feature_measure must be 'log' or 'linear'.")

    ell1 = (
        logarithm(store.m1, log_base)
        if np.all(store.m1 > 0.0)
        else np.full_like(store.m1, np.nan)
    )
    ell2 = (
        logarithm(store.m2, log_base)
        if np.all(store.m2 > 0.0)
        else np.full_like(store.m2, np.nan)
    )
    measure1 = ell1 if feature_measure == "log" else store.m1
    measure2 = ell2 if feature_measure == "log" else store.m2
    weight1 = quadrature_weights(measure1)
    weight2 = quadrature_weights(measure2)
    geometry1 = ell1 if geometry == "log" else store.m1.copy()
    geometry2 = ell2 if geometry == "log" else store.m2.copy()
    quadrature_weights(geometry1)
    quadrature_weights(geometry2)
    input_weight1 = quadrature_weights(
        ell1 if input_density_measure == "log" else store.m1
    )
    input_weight2 = quadrature_weights(
        ell2 if input_density_measure == "log" else store.m2
    )
    ordered_domain = bool(
        np.all(~store.mask | (store.m2[None, :] < store.m1[:, None]))
    )

    grid = Grid(
        m1=store.m1.copy(),
        m2=store.m2.copy(),
        ell1=ell1,
        ell2=ell2,
        geometry1=geometry1,
        geometry2=geometry2,
        mask=store.mask.copy(),
        weight1=weight1,
        weight2=weight2,
        log_base=log_base,
        geometry=geometry,
        feature_measure=feature_measure,
        input_density_measure=input_density_measure,
        domain=input_settings.domain,
        ordered_domain=ordered_domain,
        coordinate1_name=input_settings.coordinate1_name,
        coordinate2_name=input_settings.coordinate2_name,
        coordinate1_label=input_settings.coordinate1_label,
        coordinate2_label=input_settings.coordinate2_label,
        coordinate1_unit=input_settings.coordinate1_unit,
        coordinate2_unit=input_settings.coordinate2_unit,
        input_weight1=input_weight1,
        input_weight2=input_weight2,
    )

    indices = np.unique(
        np.linspace(
            0, store.number_draws - 1, min(sample_draws, store.number_draws)
        ).astype(int)
    )
    for index in indices:
        field = np.asarray(store.draws[index], dtype=float)
        valid = field[grid.mask]
        if np.any(~np.isfinite(valid)) or np.any(valid < 0.0):
            raise ValueError(f"Draw {index} contains invalid density values.")
        if not np.any(valid > 0.0):
            raise ValueError(f"Draw {index} has zero probability everywhere.")

    return grid


def log_to_feature_jacobian(grid: Grid) -> np.ndarray:
    """Jacobian dividing a per-log input density into the feature measure."""

    if grid.feature_measure == "log":
        return np.ones(grid.shape, dtype=float)
    return (
        np.multiply.outer(grid.m1, grid.m2)
        * float(np.log(grid.log_base) ** 2)
    )


def input_to_feature_factor(grid: Grid) -> np.ndarray:
    """Multiplicative conversion from the declared input to feature measure."""

    if grid.input_density_measure == grid.feature_measure:
        return np.ones(grid.shape, dtype=float)
    jacobian = (
        np.multiply.outer(grid.m1, grid.m2)
        * float(np.log(grid.log_base) ** 2)
    )
    if grid.input_density_measure == "linear":
        return jacobian
    return 1.0 / jacobian


def density_in_feature_measure(density: np.ndarray, grid: Grid) -> np.ndarray:
    """Convert values from the declared input measure to the feature measure."""

    values = np.asarray(density, dtype=float)
    if values.shape[-2:] != grid.shape:
        raise ValueError("Density shape does not agree with the grid.")
    converted = values * input_to_feature_factor(grid)
    return np.where(grid.mask, converted, 0.0)


def normalize_density(density: np.ndarray, grid: Grid) -> tuple[np.ndarray, float]:
    values = np.where(grid.mask, np.asarray(density, dtype=float), 0.0)
    integral = grid.integrate(values)
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("Density has a nonpositive or nonfinite integral.")
    return values / integral, integral


def write_hdf5(
    path: str | Path,
    p: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    h0: np.ndarray | None = None,
    external_parameter: np.ndarray | None = None,
    chain_id: np.ndarray | None = None,
    log_base: float = 10.0,
    density_measure: str = "log",
    density_dataset: str = "p",
    coordinate1_dataset: str = "m1",
    coordinate2_dataset: str = "m2",
    mask_dataset: str = "mask",
    weights_dataset: str = "weights",
    external_parameter_dataset: str = "h0",
    chain_id_dataset: str = "chain_id",
    chunks: tuple[int, int, int] | None = None,
) -> Path:
    """Write a configurable v2 input file while preserving v0.7 defaults."""

    if h5py is None:
        raise RuntimeError("write_hdf5 requires h5py.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(p)
    if values.ndim != 3:
        raise ValueError("p must have shape (draw, m1, m2).")
    if density_measure not in {"linear", "log"}:
        raise ValueError("density_measure must be 'linear' or 'log'.")
    if h0 is not None and external_parameter is not None:
        raise ValueError("Supply h0 or external_parameter, not both.")
    parameter = h0 if external_parameter is None else external_parameter
    if chunks is None:
        draw_chunk = max(1, min(16, values.shape[0]))
        chunks = (
            draw_chunk,
            min(32, values.shape[1]),
            min(32, values.shape[2]),
        )
    with h5py.File(destination, "w") as output:
        output.create_dataset(
            density_dataset,
            data=values,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        output.create_dataset(coordinate1_dataset, data=np.asarray(m1, dtype=float))
        output.create_dataset(coordinate2_dataset, data=np.asarray(m2, dtype=float))
        if mask is not None:
            output.create_dataset(mask_dataset, data=np.asarray(mask, dtype=bool))
        if weights is not None:
            output.create_dataset(
                weights_dataset, data=np.asarray(weights, dtype=float)
            )
        if parameter is not None:
            output.create_dataset(
                external_parameter_dataset, data=np.asarray(parameter, dtype=float)
            )
        if chain_id is not None:
            output.create_dataset(chain_id_dataset, data=np.asarray(chain_id))
        output.attrs["log_base"] = float(log_base)
        output.attrs["density_measure"] = density_measure
        output.attrs["schema_version"] = "2.0"
    return destination


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
