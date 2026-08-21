"""Small, strict INI configuration layer."""

from __future__ import annotations

import math
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


def _boolean(parser: ConfigParser, section: str, option: str, default: bool) -> bool:
    if not parser.has_option(section, option):
        return default
    return parser.getboolean(section, option)


def _auto_integer(value: str, *, minimum: int = 1) -> int | None:
    if value.strip().lower() == "auto":
        return None
    result = int(value)
    if result < minimum:
        raise ValueError(f"Expected an integer >= {minimum}, received {result}.")
    return result


def _log_base(value: str) -> float:
    cleaned = value.strip().lower()
    if cleaned in {"e", "natural", "ln"}:
        return math.e
    result = float(cleaned)
    if not math.isfinite(result) or result <= 0.0 or math.isclose(result, 1.0):
        raise ValueError("log_base must be e or a positive number other than 1.")
    return result


def _scales(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("At least one analysis scale is required.")
    if any((not math.isfinite(scale) or scale < 0.0) for scale in result):
        raise ValueError("Analysis scales must be finite and nonnegative.")
    return result


@dataclass(frozen=True)
class InputSettings:
    file: Path
    density_dataset: str = "p"
    coordinate1_dataset: str = "m1"
    coordinate2_dataset: str = "m2"
    mask_dataset: str = "mask"
    weights_dataset: str = "weights"
    domain: str = "legacy"
    density_measure: str = "log"
    coordinate1_name: str = "m1"
    coordinate2_name: str = "m2"
    coordinate1_label: str = r"$m_1$"
    coordinate2_label: str = r"$m_2$"
    coordinate1_unit: str = ""
    coordinate2_unit: str = ""


@dataclass(frozen=True)
class AnalysisSettings:
    run: str = "2d"
    geometry: str = "log"
    log_base: float = 10.0
    feature_measure: str = "log"
    credible_mass: float = 0.90
    scales: tuple[float, ...] = (0.0, 0.025, 0.05)
    persistence_threshold: str | float = "auto"
    persistence_gap_min_log: float = 1.0
    hessian_refinement: bool = True
    detect_plateaus: bool = True
    minimum_support: float = 0.05
    curve_points: int = 160
    support_mass: float = 0.995
    curve_smoothing_cells: float = 2.0
    ridge_probability_radius: float | None = None
    shoulder_alpha_threshold: float = 2.0
    detect_shoulders: bool = True
    ordered_tails: bool = True


@dataclass(frozen=True)
class ComputeSettings:
    workers: int | None = None
    batch_size: int | None = None
    resume: bool = True


@dataclass(frozen=True)
class AssociationSettings:
    enabled: str = "auto"
    knn: int = 5
    permutations: int = 100
    uncertainty_resamples: int = 100
    random_seed: int = 1729
    dataset: str = "h0"
    chain_id_dataset: str = "chain_id"
    parameter_name: str = "H0"
    parameter_label: str = r"$H_0$"
    parameter_unit: str = ""


@dataclass(frozen=True)
class OneDimensionalSettings:
    enabled: bool = False
    measure: str = "log"
    density_floor: float = 1.0e-5
    scales: tuple[float, ...] = (0.050, 0.075, 0.100)
    minimum_persistence: int = 2
    minimum_separation_lnm: float = 0.050
    match_tolerance_lnm: float = 0.100
    upper_tail_percentile: float = 0.999


@dataclass(frozen=True)
class OutputSettings:
    directory: Path
    overwrite: bool = False
    profile: str = "essential"
    full_subdirectory: str = "full"


@dataclass(frozen=True)
class PlotSettings:
    joint_cmap: str = "magma"
    display_probability: float = 0.995
    hpd_contours: tuple[float, ...] = (0.50, 0.90)
    write_pdf: bool = True
    write_png: bool = False


@dataclass(frozen=True)
class Settings:
    source: Path
    input: InputSettings
    analysis: AnalysisSettings
    compute: ComputeSettings
    one_dimensional: OneDimensionalSettings
    association: AssociationSettings
    output: OutputSettings
    plot: PlotSettings


def load_settings(path: str | Path) -> Settings:
    """Load settings and resolve file paths relative to the INI file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Settings file not found: {source}")

    parser = ConfigParser()
    with source.open("r", encoding="utf-8") as stream:
        parser.read_file(stream)

    if not parser.has_option("input", "file"):
        raise ValueError("[input] file is required.")
    if not parser.has_option("output", "directory"):
        raise ValueError("[output] directory is required.")

    base = source.parent

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (
            (base / candidate).resolve() if not candidate.is_absolute() else candidate
        )

    def text(section: str, option: str, fallback: str) -> str:
        value = parser.get(section, option, fallback=fallback).strip()
        if not value:
            raise ValueError(f"{section}.{option} must not be empty.")
        return value

    domain = text("input", "domain", "legacy").lower()
    if domain not in {"legacy", "full", "ordered", "mask"}:
        raise ValueError(
            "input.domain must be 'full', 'ordered', 'mask', or 'legacy'."
        )
    input_density_measure = text("input", "density_measure", "log").lower()
    if input_density_measure not in {"linear", "log"}:
        raise ValueError("input.density_measure must be 'linear' or 'log'.")

    geometry = parser.get("analysis", "geometry", fallback="log").strip().lower()
    if geometry not in {"log", "linear"}:
        raise ValueError("analysis.geometry must be 'log' or 'linear'.")

    run_mode = parser.get("analysis", "run", fallback="2d").strip().lower()
    run_aliases = {
        "1": "1d",
        "1d": "1d",
        "one-dimensional": "1d",
        "2": "2d",
        "2d": "2d",
        "two-dimensional": "2d",
        "both": "both",
    }
    if run_mode not in run_aliases:
        raise ValueError("analysis.run must be '1d', '2d', or 'both'.")
    run_mode = run_aliases[run_mode]

    feature_measure = parser.get(
        "analysis", "feature_measure", fallback="log"
    ).strip().lower()
    if feature_measure not in {"log", "linear"}:
        raise ValueError("analysis.feature_measure must be 'log' or 'linear'.")

    credible_mass = parser.getfloat("analysis", "credible_mass", fallback=0.90)
    minimum_support = parser.getfloat("analysis", "minimum_support", fallback=0.05)
    if not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must lie strictly between zero and one.")
    if not 0.0 <= minimum_support <= 1.0:
        raise ValueError("minimum_support must lie between zero and one.")

    threshold_text = parser.get(
        "analysis", "persistence_threshold", fallback="auto"
    ).strip()
    threshold: str | float
    if threshold_text.lower() == "auto":
        threshold = "auto"
    else:
        threshold = float(threshold_text)
        if threshold < 0.0 or not math.isfinite(threshold):
            raise ValueError("persistence_threshold must be nonnegative or 'auto'.")

    persistence_gap_min_log = parser.getfloat(
        "analysis", "persistence_gap_min_log", fallback=1.0
    )
    if not math.isfinite(persistence_gap_min_log) or persistence_gap_min_log <= 0.0:
        raise ValueError(
            "analysis.persistence_gap_min_log must be finite and positive."
        )

    contours = tuple(
        float(item.strip())
        for item in parser.get("plot", "hpd_contours", fallback="0.50, 0.90").split(",")
        if item.strip()
    )
    if not contours or any(not 0.0 < value < 1.0 for value in contours):
        raise ValueError("Every HPD contour probability must lie between zero and one.")
    display_probability = parser.getfloat("plot", "display_probability", fallback=0.995)
    if not 0.0 < display_probability < 1.0:
        raise ValueError("plot.display_probability must lie between zero and one.")

    workers = _auto_integer(parser.get("compute", "workers", fallback="auto"))
    batch_size = _auto_integer(parser.get("compute", "batch_size", fallback="auto"))
    curve_points = parser.getint("analysis", "curve_points", fallback=160)
    if curve_points < 16:
        raise ValueError("curve_points must be at least 16.")

    support_mass = parser.getfloat("analysis", "support_mass", fallback=0.995)
    if not 0.0 < support_mass < 1.0:
        raise ValueError("analysis.support_mass must lie strictly between zero and one.")
    curve_smoothing_cells = parser.getfloat(
        "analysis", "curve_smoothing_cells", fallback=2.0
    )
    if not math.isfinite(curve_smoothing_cells) or curve_smoothing_cells < 0.0:
        raise ValueError("analysis.curve_smoothing_cells must be finite and nonnegative.")

    radius_text = parser.get(
        "analysis", "ridge_probability_radius", fallback="auto"
    ).strip()
    radius = None if radius_text.lower() == "auto" else float(radius_text)
    if radius is not None and (not math.isfinite(radius) or radius <= 0.0):
        raise ValueError("ridge_probability_radius must be positive or 'auto'.")

    shoulder_alpha_threshold = parser.getfloat(
        "analysis", "shoulder_alpha_threshold", fallback=2.0
    )
    if not math.isfinite(shoulder_alpha_threshold):
        raise ValueError("analysis.shoulder_alpha_threshold must be finite.")
    detect_shoulders = _boolean(
        parser, "analysis", "detect_shoulders", domain == "legacy"
    )
    ordered_tails = _boolean(
        parser, "analysis", "ordered_tails", domain == "legacy"
    )

    association_knn = parser.getint("association", "knn", fallback=5)
    association_enabled = parser.get(
        "association", "enabled", fallback="auto"
    ).strip().lower()
    if association_enabled not in {"auto", "on", "off"}:
        raise ValueError("association.enabled must be 'auto', 'on', or 'off'.")
    association_permutations = parser.getint(
        "association", "permutations", fallback=100
    )
    association_resamples = parser.getint(
        "association", "uncertainty_resamples", fallback=100
    )
    if association_knn < 1:
        raise ValueError("association.knn must be at least 1.")
    if association_permutations < 0:
        raise ValueError("association.permutations must be nonnegative.")
    if association_resamples < 0:
        raise ValueError("association.uncertainty_resamples must be nonnegative.")

    one_d_measure = parser.get(
        "one_dimensional", "measure", fallback=feature_measure
    ).strip().lower()
    if one_d_measure not in {"log", "linear"}:
        raise ValueError("one_dimensional.measure must be 'log' or 'linear'.")
    one_d_floor = parser.getfloat(
        "one_dimensional", "density_floor", fallback=1.0e-5
    )
    if not math.isfinite(one_d_floor) or one_d_floor <= 0.0:
        raise ValueError("one_dimensional.density_floor must be positive.")
    one_d_scales = _scales(
        parser.get(
            "one_dimensional", "scales", fallback="0.050, 0.075, 0.100"
        )
    )
    if any(scale <= 0.0 for scale in one_d_scales):
        raise ValueError("one_dimensional.scales must be strictly positive.")
    one_d_persistence = parser.getint(
        "one_dimensional", "minimum_persistence", fallback=2
    )
    if not 1 <= one_d_persistence <= len(one_d_scales):
        raise ValueError(
            "one_dimensional.minimum_persistence must lie between 1 and the "
            "number of one-dimensional scales."
        )
    one_d_separation = parser.getfloat(
        "one_dimensional", "minimum_separation_lnm", fallback=0.050
    )
    one_d_tolerance = parser.getfloat(
        "one_dimensional", "match_tolerance_lnm", fallback=0.100
    )
    if not math.isfinite(one_d_separation) or one_d_separation <= 0.0:
        raise ValueError(
            "one_dimensional.minimum_separation_lnm must be positive."
        )
    if not math.isfinite(one_d_tolerance) or one_d_tolerance <= 0.0:
        raise ValueError("one_dimensional.match_tolerance_lnm must be positive.")
    one_d_tail = parser.getfloat(
        "one_dimensional", "upper_tail_percentile", fallback=0.999
    )
    if not 0.0 < one_d_tail < 1.0:
        raise ValueError(
            "one_dimensional.upper_tail_percentile must lie between zero and one."
        )

    one_d_enabled = _boolean(
        parser, "one_dimensional", "enabled", run_mode in {"1d", "both"}
    )
    output_profile = text("output", "profile", "essential").lower()
    if output_profile not in {"essential", "full"}:
        raise ValueError("output.profile must be 'essential' or 'full'.")
    full_subdirectory = text("output", "full_subdirectory", "full")
    if Path(full_subdirectory).name != full_subdirectory:
        raise ValueError("output.full_subdirectory must be one directory name.")

    return Settings(
        source=source,
        input=InputSettings(
            file=resolve(parser.get("input", "file")),
            density_dataset=text("input", "density_dataset", "p"),
            coordinate1_dataset=text("input", "coordinate1_dataset", "m1"),
            coordinate2_dataset=text("input", "coordinate2_dataset", "m2"),
            mask_dataset=text("input", "mask_dataset", "mask"),
            weights_dataset=text("input", "weights_dataset", "weights"),
            domain=domain,
            density_measure=input_density_measure,
            coordinate1_name=text("input", "coordinate1_name", "m1"),
            coordinate2_name=text("input", "coordinate2_name", "m2"),
            coordinate1_label=text("input", "coordinate1_label", r"$m_1$"),
            coordinate2_label=text("input", "coordinate2_label", r"$m_2$"),
            coordinate1_unit=parser.get(
                "input", "coordinate1_unit", fallback=""
            ).strip(),
            coordinate2_unit=parser.get(
                "input", "coordinate2_unit", fallback=""
            ).strip(),
        ),
        analysis=AnalysisSettings(
            run=run_mode,
            geometry=geometry,
            log_base=_log_base(parser.get("analysis", "log_base", fallback="10")),
            feature_measure=feature_measure,
            credible_mass=credible_mass,
            scales=_scales(
                parser.get("analysis", "scales", fallback="0.0, 0.025, 0.05")
            ),
            persistence_threshold=threshold,
            persistence_gap_min_log=persistence_gap_min_log,
            hessian_refinement=_boolean(parser, "analysis", "hessian_refinement", True),
            detect_plateaus=_boolean(parser, "analysis", "detect_plateaus", True),
            minimum_support=minimum_support,
            curve_points=curve_points,
            support_mass=support_mass,
            curve_smoothing_cells=curve_smoothing_cells,
            ridge_probability_radius=radius,
            shoulder_alpha_threshold=shoulder_alpha_threshold,
            detect_shoulders=detect_shoulders,
            ordered_tails=ordered_tails,
        ),
        compute=ComputeSettings(
            workers=workers,
            batch_size=batch_size,
            resume=_boolean(parser, "compute", "resume", True),
        ),
        one_dimensional=OneDimensionalSettings(
            enabled=one_d_enabled,
            measure=one_d_measure,
            density_floor=one_d_floor,
            scales=one_d_scales,
            minimum_persistence=one_d_persistence,
            minimum_separation_lnm=one_d_separation,
            match_tolerance_lnm=one_d_tolerance,
            upper_tail_percentile=one_d_tail,
        ),
        association=AssociationSettings(
            enabled=association_enabled,
            knn=association_knn,
            permutations=association_permutations,
            uncertainty_resamples=association_resamples,
            random_seed=parser.getint(
                "association", "random_seed", fallback=1729
            ),
            dataset=text("association", "dataset", "h0"),
            chain_id_dataset=text(
                "association", "chain_id_dataset", "chain_id"
            ),
            parameter_name=text("association", "parameter_name", "H0"),
            parameter_label=text(
                "association", "parameter_label", r"$H_0$"
            ),
            parameter_unit=parser.get(
                "association", "parameter_unit", fallback=""
            ).strip(),
        ),
        output=OutputSettings(
            directory=resolve(parser.get("output", "directory")),
            overwrite=_boolean(parser, "output", "overwrite", False),
            profile=output_profile,
            full_subdirectory=full_subdirectory,
        ),
        plot=PlotSettings(
            joint_cmap=parser.get("plot", "joint_cmap", fallback="magma"),
            display_probability=display_probability,
            hpd_contours=contours,
            write_pdf=_boolean(parser, "plot", "write_pdf", True),
            # Version 0.8 deliberately emits publication figures only as PDF.
            # The old option is still accepted so existing INI files parse.
            write_png=False,
        ),
    )
