"""Legacy-policy scientific regression against the unmodified 0.8.6 nodal pipeline.

The compact golden was generated from the attached 0.8.6 source, rather than
from the finite-volume implementation. It covers every numerical HDF5 product,
numeric notebook state, scientific CSV field, and exported NPZ array from one
deterministic combined 2D/1D run. Version, provenance, logs, and signatures are
deliberately absent. Rebuilding the golden requires the original 0.8.6 source.
"""

from __future__ import annotations

import csv
import gzip
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - the integration extra supplies h5py.
    h5py = None


def _write_fixture(directory: Path, placement: str = "", feature_measure: str = "log") -> Path:
    """A geometric inclusive triangle with two moving concentrations and H0."""
    masses = np.geomspace(1.0, 100.0, 41)
    log_mass = np.log10(masses)
    mask = masses[None, :] <= masses[:, None]
    spacing = np.diff(log_mass)
    weights = np.empty(masses.size)
    weights[0], weights[-1] = spacing[0] / 2, spacing[-1] / 2
    weights[1:-1] = (spacing[:-1] + spacing[1:]) / 2
    x, y = np.meshgrid(log_mass, log_mass, indexing="ij")
    shifts = np.linspace(-0.04, 0.04, 40)
    draws = []
    for shift in shifts:
        field = (
            np.exp(-0.5 * (((x - 0.62 - shift) / 0.13) ** 2 + ((y - 0.22) / 0.12) ** 2))
            + 0.75
            * np.exp(-0.5 * (((x - 1.52 + 0.3 * shift) / 0.16) ** 2 + ((y - 0.84) / 0.15) ** 2))
            + 2.0e-5
        )
        field = np.where(mask, field, 0.0)
        field /= np.sum(field * weights[:, None] * weights[None, :])
        draws.append(field)
    np.savez(
        directory / "posterior.npz",
        p=np.asarray(draws),
        m1=masses,
        m2=masses.copy(),
        mask=mask,
        h0=75.0 + 240.0 * shifts,
        chain_id=np.repeat((0, 1), 20),
        log_base=10.0,
    )
    settings = directory / "settings.ini"
    settings.write_text(
        """[input]
file = posterior.npz
domain = ordered
density_measure = log
""" + placement + """
[analysis]
run = both
geometry = log
log_base = 10
feature_measure = log
scales = 0.0, 0.025, 0.05
persistence_threshold = 0.02
hessian_refinement = false
detect_plateaus = false
curve_points = 16
ordered_tails = true
matching_policy = legacy
ridge_region_policy = legacy_two_sided

[one_dimensional]
measure = log
scales = 0.05, 0.075, 0.10
minimum_persistence = 2
upper_tail_percentile = 0.99

[compute]
workers = 1
batch_size = 20
resume = true

[association]
knn = 3
permutations = 2
uncertainty_resamples = 2
random_seed = 8

[output]
directory = result
profile = full

[plot]
write_pdf = false
write_png = false
""",
        encoding="utf-8",
    )
    if feature_measure != "log":
        settings.write_text(
            settings.read_text(encoding="utf-8")
            .replace("feature_measure = log", "feature_measure = linear")
            .replace("[one_dimensional]\nmeasure = log", "[one_dimensional]\nmeasure = linear"),
            encoding="utf-8",
        )
    return settings


def _snapshot(directory: Path) -> dict[str, np.ndarray]:
    """Collect values; metadata-free keys let new metadata coexist with old data."""
    result: dict[str, np.ndarray] = {}

    def add(key: str, value: object, *, text: bool = False) -> None:
        array = np.asarray(value)
        if array.dtype.kind in "biufc":
            result[key] = array
        elif text and array.dtype.kind in "US":
            result[key] = array.astype(str)

    def flatten(key: str, value: object) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                flatten(f"{key}/{name}", child)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                flatten(f"{key}/{index}", child)
        else:
            add(key, value)

    for path in sorted(directory.rglob("results.h5")):
        with h5py.File(path, "r") as handle:
            def visit(name: str, obj: object) -> None:
                if isinstance(obj, h5py.Dataset):
                    add(f"h5/{name}", obj[...], text=True)
                for attribute, value in obj.attrs.items():
                    add(f"h5/{name}/@{attribute}", value)
            visit("", handle)
            handle.visititems(visit)
    for path in sorted(directory.rglob("*.pkl.gz")):
        if "state" not in path.name:
            continue
        with gzip.open(path, "rb") as stream:
            flatten(f"state/{path.name}", pickle.load(stream))
    for path in sorted(directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                add(f"npz/{path.name}/{name}", archive[name], text=True)
    for path in sorted(directory.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            for column in reader.fieldnames or ():
                values = [row[column] for row in rows]
                try:
                    array = np.asarray(values, dtype=float)
                except (TypeError, ValueError):
                    array = np.asarray(values, dtype=str)
                add(f"csv/{path.name}/{column}", array, text=True)
    return result


@unittest.skipIf(h5py is None, "h5py is required for the full nodal regression")
class NodalOptInTests(unittest.TestCase):
    def test_full_scientific_outputs_match_unmodified_086(self) -> None:
        for feature_measure in ("log", "linear"):
            with self.subTest(feature_measure=feature_measure):
                self._check_measure(feature_measure)

    def _check_measure(self, feature_measure: str) -> None:
        golden_path = Path(__file__).parent / "data" / f"nodal_086_{feature_measure}_science.npz"
        with np.load(golden_path, allow_pickle=False) as archive:
            expected = {name: archive[name] for name in archive.files}
        # Ensure the golden exercises the required independent code paths.
        for prefix in ("h5/features/", "h5/global_tail/", "state/", "csv/one_two_dimensional_", "csv/external_parameter_associations.csv/"):
            self.assertTrue(any(key.startswith(prefix) for key in expected), prefix)
        for convention in (
            "",
            "grid_location = nodes\ngrid_measure = linear\n",
            "grid_location = nodes\ngrid_measure = log\n",
        ):
            with self.subTest(convention=convention or "omitted"):
                with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
                    directory = Path(temporary)
                    settings = _write_fixture(directory, convention, feature_measure)
                    environment = dict(os.environ)
                    environment["MPLBACKEND"] = "Agg"
                    environment["OPENBLAS_NUM_THREADS"] = "1"
                    completed = subprocess.run(
                        [sys.executable, "-m", "posterior_landscape", str(settings)],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    actual = _snapshot(directory / "result")
                    self.assertFalse(set(expected) - set(actual), sorted(set(expected) - set(actual)))
                    for key, reference in expected.items():
                        if reference.dtype.kind in "US":
                            np.testing.assert_array_equal(actual[key], reference, err_msg=key)
                        else:
                            np.testing.assert_allclose(
                                actual[key], reference, rtol=1.0e-12,
                                atol=1.0e-14, equal_nan=True, err_msg=key,
                            )


if __name__ == "__main__":
    unittest.main()
