"""Generate the small density ensemble used by the README quickstart.

Run after installing posterior-landscape. This illustrative Gaussian-mixture
ensemble is not the result of a scientific inference. It needs no external data.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from posterior_landscape import write_hdf5

# Editable demonstration settings.
SEED = 1729
NUMBER_DRAWS = 64
GRID_SIZE = 48
COORDINATE_MIN = 1.0
COORDINATE_MAX = 6.0
CENTRE_SCATTER = 0.04
OUTPUT_FILE = Path(__file__).resolve().parent / "demo.h5"


def main() -> None:
    rng = np.random.default_rng(SEED)
    x1 = np.linspace(COORDINATE_MIN, COORDINATE_MAX, GRID_SIZE)
    x2 = np.linspace(COORDINATE_MIN, COORDINATE_MAX, GRID_SIZE)
    first_coordinate, second_coordinate = np.meshgrid(x1, x2, indexing="ij")
    density = np.empty((NUMBER_DRAWS, x1.size, x2.size))

    for draw in range(NUMBER_DRAWS):
        shift = rng.normal(0.0, CENTRE_SCATTER, size=4)
        fraction = rng.uniform(0.50, 0.65)
        first = np.exp(-0.5 * (
            ((first_coordinate - 2.4 - shift[0]) / 0.35) ** 2
            + ((second_coordinate - 2.2 - shift[1]) / 0.40) ** 2
        )) / (0.35 * 0.40)
        second = np.exp(-0.5 * (
            ((first_coordinate - 4.4 - shift[2]) / 0.45) ** 2
            + ((second_coordinate - 4.3 - shift[3]) / 0.35) ** 2
        )) / (0.45 * 0.35)
        density[draw] = fraction * first + (1.0 - fraction) * second
        # Normalize in the same linear nodal measure used by demo.ini.
        density[draw] /= trapezoid(
            trapezoid(density[draw], x2, axis=1), x1
        )

    write_hdf5(
        OUTPUT_FILE, density, x1, x2,
        coordinate1_dataset="x1",
        coordinate2_dataset="x2",
        density_measure="linear",
        log_base=np.e,
    )
    print(f"Wrote {NUMBER_DRAWS} illustrative density draws to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
