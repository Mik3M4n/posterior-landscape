#!/usr/bin/env python3
"""Run posterior-landscape sequentially over reconstructed diagnostic runs.

The wrapper scans one parent directory, selects exactly one cached density file
per run from its HDF5 metadata, writes a resolved posterior-landscape settings
file inside each output directory, and invokes the package with the current
Python interpreter.  After the batch it can cross-match the completed catalogues
to a baseline run and write compact model-robustness tables and figures without
rerunning posterior-landscape.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BatchError(RuntimeError):
    """A clear, user-facing batch configuration or selection error."""


DISPLAY_LABELS = {
    "baseline": "Baseline",
    "R0-4-1-long": "Baseline",
    "R0-4-2-muakpha--1.0-long": r"$\mu_{\log\alpha^{-1}}=-1.0$",
    "R0-5, cap": r"$\mu_{\log\alpha^{-1}}=-1.0$ + weight cap",
    "R1": "Weaker aspect prior",
    "R1v1": "Weaker aspect prior",
    "R1v1and3": r"Weaker aspect prior + finer $L_{\rm small}$",
    "R2": "Stronger aspect prior",
    "R3": r"Finer $L_{\rm small}$",
    "R3v2": r"Finer $L_{\rm small}$",
    "R4": r"Coarser $L_{\rm small}$",
    "R5": r"$K_{\rm DP}=10$",
    "R8": r"$K_{\rm DP}=30$",
    "R7v1": r"$K_{\rm DP}=100$",
    "R10": r"$q$-bound width $0.01$",
    "R11": r"Lower $q$-bound width $10^{-4}$",
    "R0-4-1marginal-short" : r"Marginal likelihood, $\sigma^2_{\log\mathcal{L}}<1$",
    "R0-4-1marginal-short [2]" : r"Marginal likelihood, $\sigma^2_{\log\mathcal{L}}<10$",
}

# Longest keys must win: e.g. R1v1and3 must not be labelled as R1.
DISPLAY_LABEL_KEYS = tuple(
    sorted(DISPLAY_LABELS, key=lambda value: (-len(value), value))
)
DISPLAY_ORDER_KEYS = tuple(DISPLAY_LABELS)


@dataclass(frozen=True)
class ComparisonConfig:
    enabled: bool
    baseline_substring: str
    output_directory: Path
    maximum_log_centroid_distance: float
    minimum_match_margin: float
    write_pdf: bool
    write_png: bool
    figure_dpi: int


@dataclass(frozen=True)
class BatchConfig:
    ini_path: Path
    parent_directory: Path
    run_glob: str
    requested_draws_per_chain: int
    density_basename: str
    template_settings: Path
    output_subdirectory: str
    posterior_landscape_root: Path | None
    continue_on_error: bool
    dry_run: bool
    include_substrings: tuple[str, ...]
    exclude_substrings: tuple[str, ...]
    status_csv: Path
    resolved_settings_filename: str
    comparison: ComparisonConfig


@dataclass(frozen=True)
class DensityInfo:
    path: Path
    requested_draws_per_chain: int | None
    actual_draws_per_chain: tuple[int, ...] | None
    number_draws: int
    grid: tuple[int, int]
    has_h0: bool
    cache_signature: str | None

    @property
    def actual_tag(self) -> str:
        counts = self.actual_draws_per_chain
        if not counts:
            return str(self.number_draws)
        if len(set(counts)) == 1:
            return str(counts[0])
        return "-".join(str(value) for value in counts)


@dataclass(frozen=True)
class AggregationRun:
    run_directory: Path
    output_directory: Path
    density: DensityInfo
    display_label: str
    display_order: int
    status: str
    message: str
    control_directory: Path | None = None


PRIMARY_FEATURE_TYPES = (
    "peak",
    "pit",
    "ridge",
    "valley",
    "shoulder",
    "plateau",
    "depression_floor",
)

FEATURE_TYPE_ORDER = {
    feature_type: index for index, feature_type in enumerate(PRIMARY_FEATURE_TYPES)
}

BRANCH_FEATURE_TYPES = frozenset({"ridge", "valley"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _expand_path(value: str, base_directory: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _safe_relative_subdirectory(value: str) -> str:
    value = value.strip()
    if not value:
        raise BatchError("output_subdirectory cannot be empty.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise BatchError("output_subdirectory must be a relative path without '..'.")
    return value


def load_batch_config(path: Path) -> BatchConfig:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise BatchError(f"Batch settings file not found: {path}")

    parser = configparser.ConfigParser(interpolation=None)
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if "batch" not in parser:
        raise BatchError("Batch settings must contain a [batch] section.")

    section = parser["batch"]
    base = path.parent
    for required in (
        "parent_directory",
        "draws_per_chain",
        "template_settings",
        "output_subdirectory",
    ):
        if not section.get(required, "").strip():
            raise BatchError(f"Missing [batch] {required}.")

    try:
        draws_per_chain = int(section["draws_per_chain"])
    except ValueError as exc:
        raise BatchError("draws_per_chain must be -1 or a positive integer.") from exc
    if draws_per_chain == 0 or draws_per_chain < -1:
        raise BatchError("draws_per_chain must be -1 or a positive integer.")

    parent = _expand_path(section["parent_directory"], base)
    template = _expand_path(section["template_settings"], base)
    package_root_text = section.get("posterior_landscape_root", "").strip()
    package_root = (
        _expand_path(package_root_text, base) if package_root_text else None
    )

    draw_label = "all" if draws_per_chain == -1 else str(draws_per_chain)
    status_text = section.get("status_csv", "").strip()
    status_csv = (
        _expand_path(status_text, base)
        if status_text
        else parent / f"posterior_landscape_batch_status_{draw_label}_per_chain.csv"
    )

    try:
        continue_on_error = section.getboolean("continue_on_error", fallback=True)
        dry_run = section.getboolean("dry_run", fallback=False)
    except ValueError as exc:
        raise BatchError("continue_on_error and dry_run must be true or false.") from exc

    comparison_section = parser["comparison"] if "comparison" in parser else None
    try:
        comparison_enabled = (
            comparison_section.getboolean("enabled", fallback=True)
            if comparison_section is not None
            else True
        )
        comparison_write_pdf = (
            comparison_section.getboolean("write_pdf", fallback=True)
            if comparison_section is not None
            else True
        )
        comparison_write_png = (
            comparison_section.getboolean("write_png", fallback=True)
            if comparison_section is not None
            else True
        )
    except ValueError as exc:
        raise BatchError(
            "[comparison] enabled, write_pdf, and write_png must be true or false."
        ) from exc

    comparison_output_text = (
        comparison_section.get("output_directory", "").strip()
        if comparison_section is not None
        else ""
    )
    comparison_output = (
        _expand_path(comparison_output_text, base)
        if comparison_output_text
        else parent / f"posterior_landscape_robustness_{draw_label}_per_chain"
    )
    baseline_substring = (
        comparison_section.get("baseline_substring", "R0-4-1-long").strip()
        if comparison_section is not None
        else "R0-4-1-long"
    )
    try:
        maximum_log_centroid_distance = float(
            comparison_section.get("maximum_log_centroid_distance", "0.75")
            if comparison_section is not None
            else "0.75"
        )
        minimum_match_margin = float(
            comparison_section.get("minimum_match_margin", "0.10")
            if comparison_section is not None
            else "0.10"
        )
        figure_dpi = int(
            comparison_section.get("figure_dpi", "220")
            if comparison_section is not None
            else "220"
        )
    except ValueError as exc:
        raise BatchError(
            "Invalid numeric value in [comparison]."
        ) from exc
    if maximum_log_centroid_distance <= 0.0:
        raise BatchError(
            "[comparison] maximum_log_centroid_distance must be positive."
        )
    if minimum_match_margin < 0.0:
        raise BatchError("[comparison] minimum_match_margin cannot be negative.")
    if figure_dpi <= 0:
        raise BatchError("[comparison] figure_dpi must be positive.")

    comparison = ComparisonConfig(
        enabled=comparison_enabled,
        baseline_substring=baseline_substring,
        output_directory=comparison_output.resolve(),
        maximum_log_centroid_distance=maximum_log_centroid_distance,
        minimum_match_margin=minimum_match_margin,
        write_pdf=comparison_write_pdf,
        write_png=comparison_write_png,
        figure_dpi=figure_dpi,
    )

    return BatchConfig(
        ini_path=path,
        parent_directory=parent,
        run_glob=section.get("run_glob", "*").strip() or "*",
        requested_draws_per_chain=draws_per_chain,
        density_basename=(
            section.get("density_basename", "fullpop_posterior_mass_density").strip()
            or "fullpop_posterior_mass_density"
        ),
        template_settings=template,
        output_subdirectory=_safe_relative_subdirectory(
            section["output_subdirectory"]
        ),
        posterior_landscape_root=package_root,
        continue_on_error=continue_on_error,
        dry_run=dry_run,
        include_substrings=_split_csv(section.get("include", "")),
        exclude_substrings=_split_csv(section.get("exclude", "")),
        status_csv=status_csv,
        resolved_settings_filename=(
            section.get(
                "resolved_settings_filename",
                "posterior_landscape.resolved.ini",
            ).strip()
            or "posterior_landscape.resolved.ini"
        ),
        comparison=comparison,
    )


def _decode_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _parse_actual_counts(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    value = _decode_attribute(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [item for item in value.split(",") if item.strip()]
    if isinstance(value, (int, float)):
        value = [value]
    try:
        counts = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise BatchError(f"Invalid actual_draws_per_chain metadata: {value!r}") from exc
    if not counts or any(item < 0 for item in counts):
        raise BatchError(f"Invalid actual_draws_per_chain metadata: {counts!r}")
    return counts


def _requested_from_legacy_filename(path: Path, density_basename: str) -> int | None:
    pattern = re.compile(
        rf"^{re.escape(density_basename)}_(\d+)_draws_per_chain\.h5$"
    )
    match = pattern.match(path.name)
    return int(match.group(1)) if match else None


def inspect_density_file(path: Path, density_basename: str) -> DensityInfo:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise BatchError(
            "h5py is required to select and validate density files. "
            "Run this wrapper in the posterior-landscape environment."
        ) from exc

    try:
        with h5py.File(path, "r") as handle:
            missing = {"p", "m1", "m2", "mask"} - set(handle.keys())
            if missing:
                raise BatchError(f"{path}: missing datasets {sorted(missing)}")
            if handle["p"].ndim != 3:
                raise BatchError(f"{path}: p must have shape (draw, m1, m2).")
            number_draws, n_m1, n_m2 = map(int, handle["p"].shape)
            if handle["m1"].shape != (n_m1,) or handle["m2"].shape != (n_m2,):
                raise BatchError(f"{path}: density and mass-grid shapes disagree.")
            if handle["mask"].shape != (n_m1, n_m2):
                raise BatchError(f"{path}: mask shape disagrees with the density grid.")

            requested_raw = handle.attrs.get("requested_draws_per_chain")
            requested = (
                int(_decode_attribute(requested_raw))
                if requested_raw is not None
                else _requested_from_legacy_filename(path, density_basename)
            )
            actual = _parse_actual_counts(
                handle.attrs.get("actual_draws_per_chain")
            )
            if actual is not None and sum(actual) != number_draws:
                raise BatchError(
                    f"{path}: sum(actual_draws_per_chain)={sum(actual)} "
                    f"but p contains {number_draws} draws."
                )

            has_h0 = "h0" in handle
            has_chain_id = "chain_id" in handle
            if has_h0 != has_chain_id:
                raise BatchError(f"{path}: h0 and chain_id must appear together.")
            if has_h0:
                if handle["h0"].shape != (number_draws,):
                    raise BatchError(f"{path}: h0 length does not match p.")
                if handle["chain_id"].shape != (number_draws,):
                    raise BatchError(f"{path}: chain_id length does not match p.")

            signature_raw = handle.attrs.get("cache_signature")
            signature = (
                str(_decode_attribute(signature_raw))
                if signature_raw is not None
                else None
            )
    except OSError as exc:
        raise BatchError(f"Cannot read HDF5 file {path}: {exc}") from exc

    return DensityInfo(
        path=path.resolve(),
        requested_draws_per_chain=requested,
        actual_draws_per_chain=actual,
        number_draws=number_draws,
        grid=(n_m1, n_m2),
        has_h0=has_h0,
        cache_signature=signature,
    )


def select_density_file(run_directory: Path, config: BatchConfig) -> DensityInfo:
    pattern = f"{config.density_basename}_*_draws_per_chain.h5"
    paths = sorted(path for path in run_directory.glob(pattern) if path.is_file())
    if not paths:
        raise BatchError(f"No density files matching {pattern!r}.")

    inspected = [inspect_density_file(path, config.density_basename) for path in paths]
    matches = [
        info
        for info in inspected
        if info.requested_draws_per_chain == config.requested_draws_per_chain
    ]
    if len(matches) == 1:
        return matches[0]

    available = ", ".join(
        f"{info.path.name} [requested={info.requested_draws_per_chain}]"
        for info in inspected
    )
    if not matches:
        raise BatchError(
            f"No density file has requested_draws_per_chain="
            f"{config.requested_draws_per_chain}. Available: {available}"
        )
    raise BatchError(
        f"Multiple density files have requested_draws_per_chain="
        f"{config.requested_draws_per_chain}: {available}"
    )


def discover_run_directories(config: BatchConfig) -> list[Path]:
    if not config.parent_directory.is_dir():
        raise BatchError(f"Parent directory not found: {config.parent_directory}")
    directories = sorted(
        path
        for path in config.parent_directory.glob(config.run_glob)
        if path.is_dir()
    )
    if config.include_substrings:
        directories = [
            path
            for path in directories
            if any(token in path.name for token in config.include_substrings)
        ]
    if config.exclude_substrings:
        directories = [
            path
            for path in directories
            if not any(token in path.name for token in config.exclude_substrings)
        ]
    if not directories:
        raise BatchError(
            f"No run directories match {config.run_glob!r} in "
            f"{config.parent_directory}."
        )
    return directories


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_settings_bytes(
    template_path: Path,
    density_path: Path,
    output_directory: Path,
) -> bytes:
    if not template_path.is_file():
        raise BatchError(f"Template settings file not found: {template_path}")
    parser = configparser.ConfigParser(interpolation=None)
    with template_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if "input" not in parser or "output" not in parser:
        raise BatchError(
            f"{template_path} must contain [input] and [output] sections."
        )
    parser["input"]["file"] = str(density_path.resolve())
    parser["output"]["directory"] = str(output_directory.resolve())
    stream = io.StringIO()
    parser.write(stream)
    return stream.getvalue().encode("utf-8")


def _format_output_subdirectory(
    template: str,
    run_directory: Path,
    density: DensityInfo,
    requested_draws_per_chain: int,
) -> str:
    requested_label = "all" if requested_draws_per_chain == -1 else str(
        requested_draws_per_chain
    )
    try:
        formatted = template.format(
            run_name=run_directory.name,
            draws_per_chain=requested_label,
            actual_draws_per_chain=density.actual_tag,
        )
    except KeyError as exc:
        raise BatchError(
            "Unknown output_subdirectory placeholder. Supported placeholders are "
            "{run_name}, {draws_per_chain}, and {actual_draws_per_chain}."
        ) from exc
    return _safe_relative_subdirectory(formatted)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_run_files(
    run_directory: Path,
    density: DensityInfo,
    config: BatchConfig,
) -> tuple[Path, Path]:
    relative_output = _format_output_subdirectory(
        config.output_subdirectory,
        run_directory,
        density,
        config.requested_draws_per_chain,
    )
    output_directory = (run_directory / relative_output).resolve()
    settings_path = output_directory / config.resolved_settings_filename
    provenance_path = output_directory / "posterior_landscape.batch.json"
    resolved = _resolved_settings_bytes(
        config.template_settings,
        density.path,
        output_directory,
    )
    stat = density.path.stat()
    provenance = {
        "input_file": str(density.path),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "input_cache_signature": density.cache_signature,
        "requested_draws_per_chain": density.requested_draws_per_chain,
        "actual_draws_per_chain": density.actual_draws_per_chain,
        "number_draws": density.number_draws,
        "template_settings": str(config.template_settings),
        "template_sha256": _sha256_file(config.template_settings),
        "resolved_settings_sha256": _sha256_bytes(resolved),
    }

    if provenance_path.exists():
        try:
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchError(f"Cannot validate {provenance_path}: {exc}") from exc
        protected_keys = (
            "input_file",
            "input_cache_signature",
            "resolved_settings_sha256",
        )
        changed = [key for key in protected_keys if previous.get(key) != provenance.get(key)]
        if changed:
            raise BatchError(
                f"Output directory already belongs to a different input/settings "
                f"combination ({', '.join(changed)} changed): {output_directory}. "
                "Choose a different output_subdirectory."
            )
    elif output_directory.exists() and any(output_directory.iterdir()):
        raise BatchError(
            f"Non-empty output directory has no batch provenance: {output_directory}. "
            "Choose a different output_subdirectory."
        )

    if not config.dry_run:
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_atomic(settings_path, resolved)
        provenance_bytes = (
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _write_atomic(provenance_path, provenance_bytes)
    return output_directory, settings_path


def build_execution_environment(
    package_root: Path | None,
) -> tuple[dict[str, str], Path | None]:
    environment = os.environ.copy()
    working_directory = None
    if package_root is not None:
        source_directory = package_root / "src"
        package_directory = source_directory / "posterior_landscape"
        if not package_directory.is_dir():
            raise BatchError(
                f"posterior_landscape_root does not contain "
                f"src/posterior_landscape: {package_root}"
            )
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(source_directory)
            if not existing
            else str(source_directory) + os.pathsep + existing
        )
        working_directory = package_root

    probe = subprocess.run(
        [sys.executable, "-c", "import posterior_landscape"],
        cwd=working_directory,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else ""
        raise BatchError(
            "posterior_landscape is not importable with the current Python. "
            "Set posterior_landscape_root to the repository containing src/. "
            f"{detail}"
        )
    return environment, working_directory


STATUS_FIELDS = (
    "run",
    "status",
    "return_code",
    "requested_draws_per_chain",
    "actual_draws_per_chain",
    "number_draws",
    "input_file",
    "output_directory",
    "started_utc",
    "finished_utc",
    "elapsed_seconds",
    "message",
)


def write_status_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in STATUS_FIELDS})
    os.replace(temporary, path)


def _counts_text(counts: tuple[int, ...] | None) -> str:
    return "" if counts is None else ",".join(str(value) for value in counts)


def _display_label_and_order(run_name: str) -> tuple[str, int]:
    for key in DISPLAY_LABEL_KEYS:
        if key in run_name:
            return DISPLAY_LABELS[key], DISPLAY_ORDER_KEYS.index(key)
    return run_name, len(DISPLAY_ORDER_KEYS)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_value(row: dict[str, Any] | None, name: str) -> float:
    if not row:
        return math.nan
    value = row.get(name, "")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _float_alias(
    row: dict[str, Any] | None,
    *names: str,
) -> float:
    for name in names:
        value = _float_value(row, name)
        if math.isfinite(value):
            return value
    return math.nan


def _truthy_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_status_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {
        row.get("run", ""): row
        for row in _read_csv_rows(path)
        if row.get("run", "")
    }


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolved_output_layout(
    control_directory: Path,
    settings_path: Path,
) -> tuple[configparser.ConfigParser, Path]:
    """Return saved settings and the directory containing workflow products.

    The batch control files always live in ``control_directory``.  A full
    posterior-landscape run writes its products below ``full_subdirectory``;
    an essential run writes them directly in the requested directory.
    """

    parser = configparser.ConfigParser(interpolation=None)
    with settings_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)

    profile = parser.get("output", "profile", fallback="essential").strip().lower()
    if profile == "essential":
        return parser, control_directory
    if profile != "full":
        raise BatchError(
            f"invalid [output] profile {profile!r} in {settings_path}; "
            "expected 'essential' or 'full'"
        )

    full_subdirectory = parser.get(
        "output", "full_subdirectory", fallback="full"
    ).strip()
    relative = Path(full_subdirectory)
    if not full_subdirectory or relative.name != full_subdirectory:
        raise BatchError(
            f"invalid [output] full_subdirectory {full_subdirectory!r} in "
            f"{settings_path}; expected one directory name"
        )
    return parser, control_directory / relative


def _workflow_completion_status(
    control_directory: Path,
    settings_path: Path,
) -> tuple[bool, str]:
    """Check the complete requested workflow, not merely a stale status row."""

    try:
        parser, output_directory = _resolved_output_layout(
            control_directory, settings_path
        )
    except (OSError, configparser.Error, BatchError) as exc:
        return False, f"cannot resolve output layout: {exc}"

    run_mode = parser.get("analysis", "run", fallback="2d").strip().lower()
    one_d_default = run_mode in {"1d", "both"}
    try:
        one_d_enabled = parser.getboolean(
            "one_dimensional", "enabled", fallback=one_d_default
        )
    except ValueError:
        return False, "invalid [one_dimensional] enabled setting"
    run_two_d = run_mode in {"2d", "both"}

    missing: list[str] = []
    if one_d_enabled:
        for name in (
            "one_dimensional_manifest.json",
            "one_dimensional_m2_manifest.json",
        ):
            manifest = _read_json_dict(output_directory / name)
            if not manifest or not manifest.get("completed_core"):
                missing.append(name)
    if run_two_d:
        manifest = _read_json_dict(output_directory / "manifest.json")
        if not manifest or not manifest.get("completed"):
            missing.append("manifest.json")
        if not (output_directory / "results.h5").is_file():
            missing.append("results.h5")
        if not (
            (output_directory / "features.csv").is_file()
            or (output_directory / "posterior_intervals.csv").is_file()
        ):
            missing.append("features.csv")
    if one_d_enabled and run_two_d:
        comparison_manifest = _read_json_dict(
            output_directory / "one_two_dimensional_manifest.json"
        )
        if not comparison_manifest or not comparison_manifest.get("completed"):
            missing.append("one_two_dimensional_manifest.json")
        for name in (
            "one_two_dimensional_feature_comparison.csv",
            "one_two_dimensional_m2_feature_comparison.csv",
            "one_two_dimensional_feature_families.csv",
        ):
            if not (output_directory / name).is_file():
                missing.append(name)

    if missing:
        return False, "missing or incomplete: " + ", ".join(dict.fromkeys(missing))
    return True, "complete workflow products are present"


def collect_aggregation_runs(config: BatchConfig) -> list[AggregationRun]:
    status_lookup = _read_status_lookup(config.status_csv)
    runs: list[AggregationRun] = []
    for run_directory in discover_run_directories(config):
        try:
            density = select_density_file(run_directory, config)
        except Exception as exc:
            print(f"Aggregation skips {run_directory.name}: {exc}")
            continue
        relative_output = _format_output_subdirectory(
            config.output_subdirectory,
            run_directory,
            density,
            config.requested_draws_per_chain,
        )
        control_directory = (run_directory / relative_output).resolve()
        settings_path = control_directory / config.resolved_settings_filename
        status_row = status_lookup.get(run_directory.name, {})
        recorded_status = status_row.get("status", "")
        recorded_message = status_row.get("message", "")
        complete, completion_message = _workflow_completion_status(
            control_directory,
            settings_path,
        )
        try:
            _, output_directory = _resolved_output_layout(
                control_directory, settings_path
            )
        except (OSError, configparser.Error, BatchError):
            output_directory = control_directory
        if complete:
            status = (
                recorded_status
                if recorded_status in {"completed", "completed_existing"}
                else "completed_existing"
            )
            message = recorded_message
        elif recorded_status:
            status = recorded_status
            message = recorded_message or completion_message
            if recorded_status in {"completed", "completed_existing"}:
                status = "incomplete_output"
        else:
            status = "missing_output"
            message = completion_message
        label, order = _display_label_and_order(run_directory.name)
        runs.append(
            AggregationRun(
                run_directory=run_directory,
                output_directory=output_directory,
                density=density,
                display_label=label,
                display_order=order,
                status=status,
                message=message,
                control_directory=control_directory,
            )
        )
    return sorted(
        runs,
        key=lambda run: (run.display_order, run.display_label, run.run_directory.name),
    )


def _select_baseline_run(
    runs: list[AggregationRun], comparison: ComparisonConfig
) -> AggregationRun:
    usable = [
        run
        for run in runs
        if run.status in {"completed", "completed_existing"}
        and (run.output_directory / "results.h5").is_file()
    ]
    direct = [
        run
        for run in usable
        if comparison.baseline_substring
        and comparison.baseline_substring in run.run_directory.name
    ]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise BatchError(
            "[comparison] baseline_substring matches more than one completed run: "
            + ", ".join(run.run_directory.name for run in direct)
        )
    fallback = [
        run
        for run in usable
        if run.display_label == "Baseline"
        or "baseline" in run.run_directory.name.lower()
        or "_base_" in run.run_directory.name.lower()
    ]
    if len(fallback) == 1:
        return fallback[0]
    raise BatchError(
        "Cannot identify exactly one completed baseline output. Set "
        "[comparison] baseline_substring to a unique folder-name substring."
    )


def _consolidated_feature_candidate(row: dict[str, str]) -> dict[str, Any] | None:
    """Read one primary morphology from the consolidated catalogue schema.

    The batch comparison deliberately follows the catalogue emitted by each run.
    Paired saddle events remain available in the per-run products, but are not
    primary comparison rows because they are derived from the individual arms.
    """

    feature_type = row.get("type", "").strip().lower()
    if feature_type not in PRIMARY_FEATURE_TYPES:
        return None

    if feature_type == "shoulder":
        p_morph_name, p_loc_name, p_reg_name = "P_morph", "P_loc", "P_reg"
        probability_name = "region_probability_median"
        strength_name, strength_column = "slope_contrast", "slope_contrast_median"
    elif feature_type in BRANCH_FEATURE_TYPES:
        p_morph_name, p_loc_name, p_reg_name = (
            "support",
            "location_support",
            "geometry_support",
        )
        probability_name = "feature_probability_median"
        strength_name, strength_column = "log_contrast", "log_contrast_median"
    elif feature_type in {"peak", "pit"}:
        p_morph_name, p_loc_name, p_reg_name = (
            "support",
            "location_support",
            "geometry_support",
        )
        probability_name = "feature_probability_median"
        strength_name, strength_column = (
            "relative_prominence",
            "relative_prominence_median",
        )
    else:
        p_morph_name, p_loc_name, p_reg_name = (
            "support",
            "location_support",
            "geometry_support",
        )
        probability_name = "region_probability_median"
        strength_name, strength_column = "", ""

    return {
        "ID": row.get("ID", "").strip(),
        "feature_type": feature_type,
        "P_morph": _float_value(row, p_morph_name),
        "P_loc": _float_value(row, p_loc_name),
        "P_reg": _float_value(row, p_reg_name),
        "x1_lower": _float_value(row, "x1_lower"),
        "x1_median": _float_value(row, "x1_median"),
        "x1_upper": _float_value(row, "x1_upper"),
        "x2_lower": _float_value(row, "x2_lower"),
        "x2_median": _float_value(row, "x2_median"),
        "x2_upper": _float_value(row, "x2_upper"),
        "x1_reference": _float_value(row, "x1"),
        "x2_reference": _float_value(row, "x2"),
        "saddle_x1_median": _float_value(row, "saddle_x1_median"),
        "saddle_x2_median": _float_value(row, "saddle_x2_median"),
        "topology_end_x1_median": _float_value(
            row, "topology_end_x1_median"
        ),
        "topology_end_x2_median": _float_value(
            row, "topology_end_x2_median"
        ),
        "mu1_lower": _float_value(row, "mu1_lower"),
        "mu1_median": _float_value(row, "mu1_median"),
        "mu1_upper": _float_value(row, "mu1_upper"),
        "mu2_lower": _float_value(row, "mu2_lower"),
        "mu2_median": _float_value(row, "mu2_median"),
        "mu2_upper": _float_value(row, "mu2_upper"),
        "region_probability_median": _float_value(row, probability_name),
        "deficit_probability_median": _float_value(
            row, "deficit_probability_median"
        ),
        "strength_name": strength_name,
        "strength_median": _float_value(row, strength_column),
        "match_ambiguity_probability": _float_value(
            row, "match_ambiguity_probability"
        ),
        "boundary": row.get("boundary", "").strip(),
        "catalogue_status": row.get("status", "").strip(),
    }


def _positive_point(*values: float) -> tuple[float, float] | None:
    if len(values) != 2 or not all(
        math.isfinite(value) and value > 0.0 for value in values
    ):
        return None
    return float(values[0]), float(values[1])


def _candidate_point(candidate: dict[str, Any]) -> tuple[tuple[float, float] | None, str]:
    """Return the best common-coordinate point available for one feature."""

    for fields, source in (
        (("x1_median", "x2_median"), "topological_location"),
        (("mu1_median", "mu2_median"), "feature_centroid"),
        (("x1_reference", "x2_reference"), "reference_location"),
    ):
        point = _positive_point(
            *(float(candidate.get(field, math.nan)) for field in fields)
        )
        if point is not None:
            return point, source
    return None, "unavailable"


def _candidate_branch_anchors(
    candidate: dict[str, Any],
) -> tuple[tuple[tuple[float, float], tuple[float, float]] | None, str]:
    """Return a branch's saddle and endpoint in physical mass coordinates."""

    saddle = _positive_point(
        _float_value(candidate, "saddle_x1_median"),
        _float_value(candidate, "saddle_x2_median"),
    )
    endpoint = _positive_point(
        _float_value(candidate, "topology_end_x1_median"),
        _float_value(candidate, "topology_end_x2_median"),
    )
    if saddle is not None and endpoint is not None:
        return (saddle, endpoint), "saddle_and_endpoint"
    return None, "unavailable"


def _log_point_distance(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    return math.hypot(
        math.log(first[0] / second[0]), math.log(first[1] / second[1])
    )


def _candidate_match_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Type-aware distance used for all cross-run morphology matching."""

    if a.get("feature_type") != b.get("feature_type"):
        return math.inf
    if a.get("feature_type") in BRANCH_FEATURE_TYPES:
        first, _ = _candidate_branch_anchors(a)
        second, _ = _candidate_branch_anchors(b)
        if first is not None and second is not None:
            start = _log_point_distance(first[0], second[0])
            end = _log_point_distance(first[1], second[1])
            return math.sqrt(0.5 * (start**2 + end**2))
    first_point, _ = _candidate_point(a)
    second_point, _ = _candidate_point(b)
    if first_point is None or second_point is None:
        return math.inf
    return _log_point_distance(first_point, second_point)


def _candidate_match_geometry(candidate: dict[str, Any]) -> str:
    if candidate.get("feature_type") in BRANCH_FEATURE_TYPES:
        _, source = _candidate_branch_anchors(candidate)
        if source != "unavailable":
            return source
    _, source = _candidate_point(candidate)
    return source


def _h5_location_configurations(
    directory: Path,
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Recover configuration summaries retained in essential results.h5."""

    try:
        import h5py
        import numpy as np
    except ModuleNotFoundError as exc:
        raise BatchError(
            "h5py and numpy are required to read essential-profile results."
        ) from exc

    path = directory / "results.h5"
    if not path.is_file():
        return {}
    configurations: dict[str, list[dict[str, Any]]] = {}
    group_names = {
        "peak": "features",
        "pit": "features",
        "shoulder": "shoulders",
    }
    try:
        with h5py.File(path, "r") as handle:
            weights = np.asarray(handle["posterior/weights"], dtype=float)
            for candidate in candidates:
                family_id = str(candidate["ID"])
                group_name = group_names.get(str(candidate.get("feature_type", "")))
                if not group_name:
                    continue
                group_path = f"{group_name}/{family_id}"
                if group_path not in handle:
                    continue
                group = handle[group_path]
                status = str(
                    _decode_attribute(
                        group.attrs.get("location_configuration_status", "")
                    )
                )
                count = int(group.attrs.get("location_configuration_count", 0))
                if status != "robust_multimodal" or count < 2:
                    continue
                required = {
                    "location_configuration",
                    "location_configuration_probability",
                    "location_configuration_conditional_probability",
                    "mass_centroid",
                }
                if not required.issubset(group.keys()):
                    raise BatchError(
                        f"{path}: {group_path} lacks saved location-configuration arrays."
                    )
                names_text = str(
                    _decode_attribute(
                        group.attrs.get("location_configuration_names", "")
                    )
                )
                names = tuple(item for item in names_text.split(",") if item)
                if len(names) != count:
                    names = tuple(
                        chr(ord("A") + index) if index < 26 else str(index + 1)
                        for index in range(count)
                    )
                labels = np.asarray(group["location_configuration"], dtype=int)
                probabilities = np.asarray(
                    group["location_configuration_probability"], dtype=float
                )
                conditional = np.asarray(
                    group["location_configuration_conditional_probability"],
                    dtype=float,
                )
                centroids = np.asarray(group["mass_centroid"], dtype=float)
                rows: list[dict[str, Any]] = []
                for index, name in enumerate(names):
                    selected = labels == index
                    record: dict[str, Any] = {
                        "family_ID": family_id,
                        "configuration_ID": f"{family_id}-{name}",
                        "configuration": name,
                        "role": "main" if index == 0 else "alternative",
                        "P_configuration": (
                            float(probabilities[index])
                            if index < probabilities.size
                            else float(np.sum(weights[selected]))
                        ),
                        "P_configuration_given_region": (
                            float(conditional[index])
                            if index < conditional.size
                            else math.nan
                        ),
                    }
                    for coordinate, column in (("mu1", 0), ("mu2", 1)):
                        values = centroids[:, column]
                        valid = selected & np.isfinite(values) & (values > 0.0)
                        interval = _weighted_quantiles(values[valid], weights[valid])
                        for suffix, value in zip(
                            ("lower", "median", "upper"), interval
                        ):
                            record[f"{coordinate}_{suffix}"] = value
                    rows.append(record)
                configurations[family_id] = rows
    except (OSError, KeyError, ValueError) as exc:
        raise BatchError(f"Cannot read location configurations from {path}: {exc}") from exc
    return configurations


def _standard_feature_candidates(directory: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    consolidated_path = directory / "features.csv"
    if consolidated_path.is_file():
        for row in _read_csv_rows(consolidated_path):
            candidate = _consolidated_feature_candidate(row)
            if candidate is not None:
                candidates.append(candidate)
    else:
        legacy_rows = [
            *_read_csv_rows(directory / "posterior_intervals.csv"),
            *_read_csv_rows(directory / "shoulder_catalogue.csv"),
        ]
        for row in legacy_rows:
            candidate = _consolidated_feature_candidate(row)
            if candidate is not None:
                candidates.append(candidate)

    if not consolidated_path.is_file() and not legacy_rows:
        raise BatchError(
            f"No consolidated or legacy primary feature catalogue was found in "
            f"{directory}."
        )

    # Use the dominant location configuration when a family is multimodal.
    # This prevents a mixture of separated configurations from creating a
    # centroid that does not describe any actual posterior mode.
    configurations: dict[str, list[dict[str, Any]]] = {}
    for row in _read_csv_rows(directory / "feature_location_configurations.csv"):
        family_id = row.get("family_ID", "")
        if family_id:
            configurations.setdefault(family_id, []).append(row)
    if not configurations:
        configurations = _h5_location_configurations(directory, candidates)

    for candidate in candidates:
        rows = configurations.get(candidate["ID"], [])
        if not rows:
            candidate["configuration_ID"] = ""
            candidate["configuration_role"] = ""
            candidate["P_configuration"] = math.nan
            candidate["P_configuration_given_region"] = math.nan
            candidate["configuration_count"] = 0
            continue
        dominant = max(
            rows,
            key=lambda row: (
                row.get("role", "") == "main",
                _float_value(row, "P_configuration"),
            ),
        )
        candidate["configuration_ID"] = dominant.get("configuration_ID", "")
        candidate["configuration_role"] = dominant.get("role", "")
        candidate["P_configuration"] = _float_value(
            dominant, "P_configuration"
        )
        candidate["P_configuration_given_region"] = _float_value(
            dominant, "P_configuration_given_region"
        )
        candidate["configuration_count"] = len(rows)
        for coordinate in ("mu1", "mu2"):
            for quantile in ("lower", "median", "upper"):
                value = _float_value(dominant, f"{coordinate}_{quantile}")
                if math.isfinite(value):
                    candidate[f"{coordinate}_{quantile}"] = value

    return sorted(
        (candidate for candidate in candidates if candidate["ID"]),
        key=lambda item: (
            FEATURE_TYPE_ORDER.get(str(item.get("feature_type", "")), len(FEATURE_TYPE_ORDER)),
            str(item["ID"]),
        ),
    )


def _assignment_relation(
    *, split: bool, merge: bool
) -> str:
    if split and merge:
        return "split_merge"
    if split:
        return "split"
    if merge:
        return "merge"
    return "none"


def _assign_baseline_features(
    baseline_features: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    comparison: ComparisonConfig,
) -> dict[str, dict[str, Any]]:
    """One-to-one primary matches plus explicit compatible split/merge links."""

    assignments: dict[str, dict[str, Any]] = {}
    for feature_type in PRIMARY_FEATURE_TYPES:
        references = [
            item for item in baseline_features if item.get("feature_type") == feature_type
        ]
        current = [
            item for item in candidates if item.get("feature_type") == feature_type
        ]
        if not references:
            continue
        distances: dict[tuple[int, int], float] = {}
        compatible_by_reference: dict[int, list[tuple[float, int]]] = {
            index: [] for index in range(len(references))
        }
        compatible_by_candidate: dict[int, list[tuple[float, int]]] = {
            index: [] for index in range(len(current))
        }
        for reference_index, reference in enumerate(references):
            for candidate_index, candidate in enumerate(current):
                distance = _candidate_match_distance(reference, candidate)
                distances[(reference_index, candidate_index)] = distance
                if (
                    math.isfinite(distance)
                    and distance <= comparison.maximum_log_centroid_distance
                ):
                    compatible_by_reference[reference_index].append(
                        (distance, candidate_index)
                    )
                    compatible_by_candidate[candidate_index].append(
                        (distance, reference_index)
                    )

        possible = sorted(
            (distance, reference_index, candidate_index)
            for (reference_index, candidate_index), distance in distances.items()
            if math.isfinite(distance)
            and distance <= comparison.maximum_log_centroid_distance
        )
        selected: dict[int, tuple[int, float]] = {}
        used_candidates: set[int] = set()
        for distance, reference_index, candidate_index in possible:
            if reference_index in selected or candidate_index in used_candidates:
                continue
            selected[reference_index] = (candidate_index, distance)
            used_candidates.add(candidate_index)

        for reference_index, reference in enumerate(references):
            compatible = sorted(compatible_by_reference[reference_index])
            selected_match = selected.get(reference_index)
            candidate: dict[str, Any] | None = None
            distance = math.nan
            margin = math.nan
            match_status = "not_detected"
            merge = any(
                len(compatible_by_candidate[candidate_index]) > 1
                for _, candidate_index in compatible
            )
            if selected_match is not None:
                candidate_index, distance = selected_match
                candidate = current[candidate_index]
                margin = (
                    compatible[1][0] - compatible[0][0]
                    if len(compatible) > 1
                    else math.inf
                )
                match_status = (
                    "ambiguous"
                    if math.isfinite(margin)
                    and margin < comparison.minimum_match_margin
                    else "matched"
                )
            assignments[str(reference["ID"])] = {
                "candidate": candidate,
                "match_status": match_status,
                "distance": distance,
                "margin": margin,
                "structural_relation": _assignment_relation(
                    split=len(compatible) > 1,
                    merge=merge,
                ),
                "compatible_candidate_ids": tuple(
                    str(current[index]["ID"]) for _, index in compatible
                ),
            }
    return assignments


BASELINE_FEATURE_FIELDS = (
    "run",
    "display_label",
    "display_order",
    "run_status",
    "baseline_ID",
    "baseline_type",
    "baseline_boundary",
    "matched_ID",
    "matched_type",
    "matched_boundary",
    "configuration_ID",
    "configuration_role",
    "match_status",
    "structural_relation",
    "compatible_candidate_IDs",
    "matching_geometry",
    "match_distance_log",
    "match_margin_log",
    "P_morph",
    "P_loc",
    "P_reg",
    "geometry_gap",
    "P_configuration",
    "P_configuration_given_region",
    "configuration_count",
    "mu1_lower",
    "mu1_median",
    "mu1_upper",
    "mu2_lower",
    "mu2_median",
    "mu2_upper",
    "centroid_log_displacement_from_baseline",
    "centroid_approx_percent_displacement",
    "region_probability_median",
    "deficit_probability_median",
    "strength_name",
    "strength_median",
    "match_ambiguity_probability",
    "h0_morph_rrb",
    "h0_mu1_rho",
    "h0_mu2_rho",
    "h0_mu_vector_mi_bits",
    "m1_1d_ID",
    "m1_projection_capture_median",
    "m1_mass_scale_spearman",
    "m2_1d_ID",
    "m2_projection_capture_median",
    "m2_mass_scale_spearman",
    "both_marginals_common_support",
    "h0_m1_only_mi_bits",
    "h0_both_marginals_mi_bits",
    "h0_2d_vector_common_mi_bits",
    "h0_2d_given_both_marginals_cmi_bits",
)


FEATURE_FAMILY_FIELDS = (
    "family_ID",
    "feature_type",
    "representative_run",
    "representative_ID",
    "representative_boundary",
    "matching_geometry",
    "number_members",
    "number_runs_detected",
    "fraction_runs_detected",
    "baseline_present",
    "baseline_IDs",
    "member_runs",
)


FEATURE_FAMILY_MATRIX_FIELDS = (
    "family_ID",
    "feature_type",
    "run",
    "display_label",
    "display_order",
    "run_status",
    "family_status",
    "member_IDs",
    "member_count",
    "representative_distance_log",
    "P_morph",
    "P_loc",
    "P_reg",
    "boundary_values",
)


ADDITIONAL_FEATURE_FIELDS = (
    "family_ID",
    "feature_type",
    "number_members",
    "number_runs_detected",
    "fraction_runs_detected",
    "member_runs",
    "representative_run",
    "representative_ID",
    "representative_boundary",
    "nearest_baseline_ID",
    "nearest_baseline_distance_log",
)


SPLIT_MERGE_FIELDS = (
    "run",
    "display_label",
    "display_order",
    "relation_type",
    "feature_type",
    "baseline_ID",
    "candidate_ID",
    "related_IDs",
    "match_distances_log",
)


GLOBAL_SUMMARY_FIELDS = (
    "run",
    "display_label",
    "display_order",
    "run_status",
    "message",
    "number_draws",
    "actual_draws_per_chain",
    "input_file",
    "output_directory",
    "h0_lower",
    "h0_median",
    "h0_upper",
    "tail_percentile",
    "m1_scale_lower",
    "m1_scale_median",
    "m1_scale_upper",
    "m2_scale_lower",
    "m2_scale_median",
    "m2_scale_upper",
    "m1_minus_m2_lower",
    "m1_minus_m2_median",
    "m1_minus_m2_upper",
    "m1_over_m2_lower",
    "m1_over_m2_median",
    "m1_over_m2_upper",
    "both_fraction_at_m1_scale_lower",
    "both_fraction_at_m1_scale_median",
    "both_fraction_at_m1_scale_upper",
    "straddle_fraction_at_m1_scale_lower",
    "straddle_fraction_at_m1_scale_median",
    "straddle_fraction_at_m1_scale_upper",
    "h0_m1_tail_rho",
    "h0_m1_tail_mi_bits",
    "h0_m2_tail_rho",
    "h0_m2_tail_mi_bits",
    "h0_tail_vector_mi_bits",
)


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, path)


def _weighted_quantiles(
    values: Any,
    weights: Any,
    probabilities: tuple[float, ...] = (0.05, 0.50, 0.95),
) -> tuple[float, ...]:
    import numpy as np

    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return tuple(math.nan for _ in probabilities)
    values = values[usable]
    weights = weights[usable]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative = (cumulative - 0.5 * weights) / cumulative[-1]
    return tuple(
        float(np.interp(probability, cumulative, values))
        for probability in probabilities
    )


def _first_matching_row(
    rows: list[dict[str, str]],
    identifiers: tuple[str, ...],
    *,
    primary_only: bool = False,
) -> dict[str, str] | None:
    for identifier in identifiers:
        if not identifier:
            continue
        for row in rows:
            if primary_only and not _truthy_csv(row.get("primary_match", "")):
                continue
            if identifier in {
                row.get("ID", ""),
                row.get("two_d_ID", ""),
                row.get("two_d_parent_ID", ""),
                row.get("family_ID", ""),
                row.get("parent_ID", ""),
            }:
                return row
    return None


def _feature_auxiliary_rows(
    directory: Path, candidate: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(candidate.get("ID", ""))
    configuration_id = str(candidate.get("configuration_ID", ""))
    identifiers = (configuration_id, candidate_id)
    projection_identifiers = (
        (configuration_id,) if configuration_id else (candidate_id,)
    )

    associations = _read_csv_rows(
        directory / "external_parameter_associations.csv"
    )
    if associations:
        h0_rows = [
            row
            for row in associations
            if row.get("parameter_name", "") in {"", "H0"}
        ]
        associations = h0_rows or associations
    else:
        associations = _read_csv_rows(directory / "h0_feature_associations.csv")
    h0_row = _first_matching_row(associations, identifiers)

    m1_rows = _read_csv_rows(directory / "one_two_dimensional_feature_comparison.csv")
    m2_rows = _read_csv_rows(
        directory / "one_two_dimensional_m2_feature_comparison.csv"
    )
    m1_row = _first_matching_row(
        m1_rows, projection_identifiers, primary_only=True
    )
    m2_row = _first_matching_row(
        m2_rows, projection_identifiers, primary_only=True
    )

    family_rows = _read_csv_rows(directory / "one_two_dimensional_feature_families.csv")
    family_row = _first_matching_row(family_rows, (candidate_id,))

    full_rows = _read_csv_rows(
        directory / "one_two_dimensional_full_h0_comparison.csv"
    )
    full_row = _first_matching_row(full_rows, identifiers)

    return {
        "h0_morph_rrb": _float_value(h0_row, "morph_rrb"),
        "h0_mu1_rho": _float_value(h0_row, "mu1_rho"),
        "h0_mu2_rho": _float_value(h0_row, "mu2_rho"),
        "h0_mu_vector_mi_bits": _float_value(h0_row, "mu_vector_mi_bits"),
        "m1_1d_ID": m1_row.get("one_d_ID", "") if m1_row else "",
        "m1_projection_capture_median": _float_value(
            m1_row, "projection_capture_median"
        ),
        "m1_mass_scale_spearman": _float_value(m1_row, "mass_scale_spearman"),
        "m2_1d_ID": m2_row.get("one_d_ID", "") if m2_row else "",
        "m2_projection_capture_median": _float_value(
            m2_row, "projection_capture_median"
        ),
        "m2_mass_scale_spearman": _float_value(m2_row, "mass_scale_spearman"),
        "both_marginals_common_support": _float_value(
            family_row, "common_support"
        ),
        "h0_m1_only_mi_bits": (
            _float_alias(
                full_row, "h0_one_d_mi_bits", "parameter_one_d_mi_bits"
            )
            if full_row
            else _float_alias(
                m1_row, "parameter_one_d_mi_bits", "h0_one_d_mi_bits"
            )
        ),
        "h0_both_marginals_mi_bits": (
            _float_alias(
                family_row,
                "parameter_1d_pair_mi_bits",
                "h0_1d_pair_mi_bits",
            )
            if family_row
            else math.nan
        ),
        "h0_2d_vector_common_mi_bits": (
            _float_alias(
                family_row,
                "parameter_2d_vector_mi_bits",
                "h0_2d_vector_mi_bits",
            )
            if family_row
            else (
                _float_alias(
                    full_row,
                    "h0_full_2d_mi_bits",
                    "parameter_two_d_mi_bits",
                )
                if full_row
                else _float_alias(
                    m1_row, "parameter_two_d_mi_bits", "h0_two_d_mi_bits"
                )
            )
        ),
        "h0_2d_given_both_marginals_cmi_bits": _float_alias(
            family_row,
            "parameter_2d_given_1d_pair_cmi_bits",
            "h0_2d_given_1d_pair_cmi_bits",
        ),
    }


def _completed_aggregation_run(run: AggregationRun) -> bool:
    return (
        run.status in {"completed", "completed_existing"}
        and (run.output_directory / "results.h5").is_file()
    )


def _baseline_assignments(
    baseline_features: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(feature["ID"]): {
            "candidate": feature,
            "match_status": "baseline",
            "distance": 0.0,
            "margin": math.inf,
            "structural_relation": "none",
            "compatible_candidate_ids": (str(feature["ID"]),),
        }
        for feature in baseline_features
    }


def _candidate_row_values(
    candidate: dict[str, Any], run: AggregationRun
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in (
        "P_morph",
        "P_loc",
        "P_reg",
        "P_configuration",
        "P_configuration_given_region",
        "configuration_count",
        "mu1_lower",
        "mu1_median",
        "mu1_upper",
        "mu2_lower",
        "mu2_median",
        "mu2_upper",
        "region_probability_median",
        "deficit_probability_median",
        "strength_name",
        "strength_median",
        "match_ambiguity_probability",
    ):
        row[field] = candidate.get(field, math.nan)
    p_morph = _float_value(candidate, "P_morph")
    p_reg = _float_value(candidate, "P_reg")
    row["geometry_gap"] = (
        p_morph - p_reg
        if math.isfinite(p_morph) and math.isfinite(p_reg)
        else math.nan
    )
    row.update(_feature_auxiliary_rows(run.output_directory, candidate))
    return row


def _baseline_feature_rows_for_run(
    run: AggregationRun,
    baseline_features: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference in baseline_features:
        assignment = assignments.get(
            str(reference["ID"]),
            {
                "candidate": None,
                "match_status": "run_failed",
                "distance": math.nan,
                "margin": math.nan,
                "structural_relation": "none",
                "compatible_candidate_ids": (),
            },
        )
        candidate = assignment["candidate"]
        row: dict[str, Any] = {
            "run": run.run_directory.name,
            "display_label": run.display_label,
            "display_order": run.display_order,
            "run_status": run.status,
            "baseline_ID": reference["ID"],
            "baseline_type": reference.get("feature_type", ""),
            "baseline_boundary": reference.get("boundary", ""),
            "matched_ID": candidate.get("ID", "") if candidate else "",
            "matched_type": candidate.get("feature_type", "") if candidate else "",
            "matched_boundary": candidate.get("boundary", "") if candidate else "",
            "configuration_ID": candidate.get("configuration_ID", "") if candidate else "",
            "configuration_role": candidate.get("configuration_role", "") if candidate else "",
            "match_status": assignment["match_status"],
            "structural_relation": assignment["structural_relation"],
            "compatible_candidate_IDs": ";".join(
                assignment["compatible_candidate_ids"]
            ),
            "matching_geometry": _candidate_match_geometry(
                candidate if candidate is not None else reference
            ),
            "match_distance_log": assignment["distance"],
            "match_margin_log": assignment["margin"],
            "centroid_log_displacement_from_baseline": assignment["distance"],
            "centroid_approx_percent_displacement": (
                100.0 * assignment["distance"]
                if math.isfinite(assignment["distance"])
                else math.nan
            ),
        }
        if candidate is not None:
            row.update(_candidate_row_values(candidate, run))
        rows.append(row)
    return rows


def _compatibility_groups(
    baseline_features: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    comparison: ComparisonConfig,
) -> tuple[dict[str, list[tuple[float, dict[str, Any]]]], dict[str, list[tuple[float, dict[str, Any]]]]]:
    """Return all permissible baseline--candidate links, grouped both ways."""

    by_baseline = {str(feature["ID"]): [] for feature in baseline_features}
    by_candidate = {str(feature["ID"]): [] for feature in candidates}
    for reference in baseline_features:
        for candidate in candidates:
            distance = _candidate_match_distance(reference, candidate)
            if (
                math.isfinite(distance)
                and distance <= comparison.maximum_log_centroid_distance
            ):
                by_baseline[str(reference["ID"])].append((distance, candidate))
                by_candidate[str(candidate["ID"])].append((distance, reference))
    for values in (*by_baseline.values(), *by_candidate.values()):
        values.sort(key=lambda item: (item[0], str(item[1]["ID"])))
    return by_baseline, by_candidate


def _split_merge_rows_for_run(
    run: AggregationRun,
    baseline_features: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    comparison: ComparisonConfig,
) -> list[dict[str, Any]]:
    if not _completed_aggregation_run(run):
        return []
    by_baseline, by_candidate = _compatibility_groups(
        baseline_features, candidates, comparison
    )
    rows: list[dict[str, Any]] = []
    baseline_by_id = {str(feature["ID"]): feature for feature in baseline_features}
    candidate_by_id = {str(feature["ID"]): feature for feature in candidates}
    for baseline_id, links in by_baseline.items():
        if len(links) <= 1:
            continue
        reference = baseline_by_id[baseline_id]
        rows.append(
            {
                "run": run.run_directory.name,
                "display_label": run.display_label,
                "display_order": run.display_order,
                "relation_type": "split",
                "feature_type": reference.get("feature_type", ""),
                "baseline_ID": baseline_id,
                "candidate_ID": "",
                "related_IDs": ";".join(str(item["ID"]) for _, item in links),
                "match_distances_log": ";".join(
                    f"{distance:.8g}" for distance, _ in links
                ),
            }
        )
    for candidate_id, links in by_candidate.items():
        if len(links) <= 1:
            continue
        candidate = candidate_by_id[candidate_id]
        rows.append(
            {
                "run": run.run_directory.name,
                "display_label": run.display_label,
                "display_order": run.display_order,
                "relation_type": "merge",
                "feature_type": candidate.get("feature_type", ""),
                "baseline_ID": "",
                "candidate_ID": candidate_id,
                "related_IDs": ";".join(str(item["ID"]) for _, item in links),
                "match_distances_log": ";".join(
                    f"{distance:.8g}" for distance, _ in links
                ),
            }
        )
    return rows


def _feature_family_components(
    candidates_by_run: dict[str, list[dict[str, Any]]],
    comparison: ComparisonConfig,
) -> list[list[dict[str, Any]]]:
    """Build reference-free families without transitive compatibility chains."""

    nodes: list[dict[str, Any]] = []
    for run_name in sorted(candidates_by_run):
        for candidate in candidates_by_run[run_name]:
            node = dict(candidate)
            node["source_run"] = run_name
            nodes.append(node)
    clusters: list[list[dict[str, Any]]] = [[node] for node in nodes]
    threshold = comparison.maximum_log_centroid_distance

    # Complete-linkage merging prevents A--B--C chains from creating a family
    # when A and C are themselves outside the matching threshold.  Same-run
    # members remain possible when every cross-member distance is compatible.
    while True:
        possible: list[tuple[float, int, int]] = []
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                distances = [
                    _candidate_match_distance(left, right)
                    for left in clusters[first]
                    for right in clusters[second]
                ]
                if distances and all(
                    math.isfinite(distance) and distance <= threshold
                    for distance in distances
                ):
                    possible.append((max(distances), first, second))
        if not possible:
            break
        _, first, second = min(
            possible,
            key=lambda item: (
                item[0],
                str(clusters[item[1]][0]["source_run"]),
                str(clusters[item[1]][0]["ID"]),
                str(clusters[item[2]][0]["source_run"]),
                str(clusters[item[2]][0]["ID"]),
            ),
        )
        clusters[first].extend(clusters.pop(second))
    return clusters


def _family_medoid(members: list[dict[str, Any]]) -> dict[str, Any]:
    def score(member: dict[str, Any]) -> tuple[float, str, str]:
        distances = [
            _candidate_match_distance(member, other)
            for other in members
            if other is not member
        ]
        finite = [distance for distance in distances if math.isfinite(distance)]
        mean = sum(finite) / len(finite) if finite else math.inf
        return mean, str(member["source_run"]), str(member["ID"])

    return min(members, key=score)


def _feature_family_sort_key(family: dict[str, Any]) -> tuple[int, float, float, str]:
    representative = family["representative"]
    point, _ = _candidate_point(representative)
    return (
        FEATURE_TYPE_ORDER.get(str(family["feature_type"]), len(FEATURE_TYPE_ORDER)),
        point[0] if point is not None else math.inf,
        point[1] if point is not None else math.inf,
        str(representative["ID"]),
    )


def _build_feature_families(
    candidates_by_run: dict[str, list[dict[str, Any]]],
    runs: list[AggregationRun],
    baseline: AggregationRun,
    comparison: ComparisonConfig,
) -> list[dict[str, Any]]:
    components = _feature_family_components(candidates_by_run, comparison)
    provisional = [
        {
            "feature_type": members[0]["feature_type"],
            "members": members,
            "representative": _family_medoid(members),
        }
        for members in components
        if members
    ]
    provisional.sort(key=_feature_family_sort_key)
    counters: dict[str, int] = {}
    families: list[dict[str, Any]] = []
    for family in provisional:
        feature_type = str(family["feature_type"])
        counters[feature_type] = counters.get(feature_type, 0) + 1
        family["family_ID"] = f"{feature_type}_{counters[feature_type]:02d}"
        families.append(family)
    return families


def _family_summary_rows(
    families: list[dict[str, Any]],
    runs: list[AggregationRun],
    baseline: AggregationRun,
) -> list[dict[str, Any]]:
    completed_names = {
        run.run_directory.name for run in runs if _completed_aggregation_run(run)
    }
    rows: list[dict[str, Any]] = []
    for family in families:
        members = family["members"]
        run_names = sorted({str(member["source_run"]) for member in members})
        representative = family["representative"]
        baseline_members = [
            member
            for member in members
            if member["source_run"] == baseline.run_directory.name
        ]
        rows.append(
            {
                "family_ID": family["family_ID"],
                "feature_type": family["feature_type"],
                "representative_run": representative["source_run"],
                "representative_ID": representative["ID"],
                "representative_boundary": representative.get("boundary", ""),
                "matching_geometry": _candidate_match_geometry(representative),
                "number_members": len(members),
                "number_runs_detected": len(run_names),
                "fraction_runs_detected": (
                    len(run_names) / len(completed_names)
                    if completed_names
                    else math.nan
                ),
                "baseline_present": bool(baseline_members),
                "baseline_IDs": ";".join(
                    str(member["ID"]) for member in baseline_members
                ),
                "member_runs": ";".join(run_names),
            }
        )
    return rows


def _feature_family_matrix_rows(
    families: list[dict[str, Any]],
    runs: list[AggregationRun],
    baseline: AggregationRun,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in families:
        representative = family["representative"]
        for run in runs:
            members = [
                member
                for member in family["members"]
                if member["source_run"] == run.run_directory.name
            ]
            if not _completed_aggregation_run(run):
                status = "run_failed"
            elif not members:
                status = "not_detected"
            elif len(members) > 1:
                status = "split"
            elif run.run_directory.name == baseline.run_directory.name:
                status = "baseline"
            else:
                status = "detected"
            selected = (
                min(
                    members,
                    key=lambda member: _candidate_match_distance(
                        representative, member
                    ),
                )
                if members
                else None
            )
            row: dict[str, Any] = {
                "family_ID": family["family_ID"],
                "feature_type": family["feature_type"],
                "run": run.run_directory.name,
                "display_label": run.display_label,
                "display_order": run.display_order,
                "run_status": run.status,
                "family_status": status,
                "member_IDs": ";".join(str(member["ID"]) for member in members),
                "member_count": len(members),
                "representative_distance_log": (
                    _candidate_match_distance(representative, selected)
                    if selected is not None
                    else math.nan
                ),
                "boundary_values": ";".join(
                    sorted(
                        {
                            str(member.get("boundary", ""))
                            for member in members
                            if str(member.get("boundary", ""))
                        }
                    )
                ),
            }
            if selected is not None:
                for field in ("P_morph", "P_loc", "P_reg"):
                    row[field] = selected.get(field, math.nan)
            rows.append(row)
    return rows


def _additional_feature_rows(
    families: list[dict[str, Any]],
    baseline_features: list[dict[str, Any]],
    runs: list[AggregationRun],
    baseline: AggregationRun,
) -> list[dict[str, Any]]:
    completed_count = sum(_completed_aggregation_run(run) for run in runs)
    rows: list[dict[str, Any]] = []
    for family in families:
        members = family["members"]
        if any(member["source_run"] == baseline.run_directory.name for member in members):
            continue
        representative = family["representative"]
        candidates = [
            (
                _candidate_match_distance(representative, baseline_feature),
                baseline_feature,
            )
            for baseline_feature in baseline_features
            if baseline_feature.get("feature_type") == family["feature_type"]
        ]
        finite = [item for item in candidates if math.isfinite(item[0])]
        nearest_distance, nearest = (
            min(finite, key=lambda item: (item[0], str(item[1]["ID"])))
            if finite
            else (math.nan, None)
        )
        run_names = sorted({str(member["source_run"]) for member in members})
        rows.append(
            {
                "family_ID": family["family_ID"],
                "feature_type": family["feature_type"],
                "number_members": len(members),
                "number_runs_detected": len(run_names),
                "fraction_runs_detected": (
                    len(run_names) / completed_count if completed_count else math.nan
                ),
                "member_runs": ";".join(run_names),
                "representative_run": representative["source_run"],
                "representative_ID": representative["ID"],
                "representative_boundary": representative.get("boundary", ""),
                "nearest_baseline_ID": nearest.get("ID", "") if nearest else "",
                "nearest_baseline_distance_log": nearest_distance,
            }
        )
    return rows


def _global_row_for_run(
    run: AggregationRun,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row: dict[str, Any] = {
        "run": run.run_directory.name,
        "display_label": run.display_label,
        "display_order": run.display_order,
        "run_status": run.status,
        "message": run.message,
        "number_draws": run.density.number_draws,
        "actual_draws_per_chain": _counts_text(run.density.actual_draws_per_chain),
        "input_file": str(run.density.path),
        "output_directory": str(run.output_directory),
    }
    if not _completed_aggregation_run(run):
        return row, None

    try:
        import h5py
        import numpy as np
    except ModuleNotFoundError as exc:
        raise BatchError(
            "h5py and numpy are required for robustness aggregation."
        ) from exc

    plot_data: dict[str, Any] | None = None
    try:
        with h5py.File(run.output_directory / "results.h5", "r") as handle:
            posterior = handle["posterior"]
            weights = np.asarray(posterior["weights"], dtype=float)
            parameter_key = (
                "h0"
                if "h0" in posterior
                else "external_parameter"
                if "external_parameter" in posterior
                else ""
            )
            parameter_values = None
            if parameter_key:
                parameter_values = np.asarray(
                    posterior[parameter_key], dtype=float
                )
                h0_lower, h0_median, h0_upper = _weighted_quantiles(
                    parameter_values, weights
                )
                row.update(
                    h0_lower=h0_lower,
                    h0_median=h0_median,
                    h0_upper=h0_upper,
                )
            if {
                "marginal1_quantiles",
                "marginal2_quantiles",
            }.issubset(posterior.keys()):
                plot_data = {
                    "m1": np.asarray(handle["coordinates/m1"], dtype=float),
                    "m2": np.asarray(handle["coordinates/m2"], dtype=float),
                    "marginal1": np.asarray(
                        posterior["marginal1_quantiles"], dtype=float
                    ),
                    "marginal2": np.asarray(
                        posterior["marginal2_quantiles"], dtype=float
                    ),
                    "weights": weights,
                }
                if parameter_values is not None:
                    plot_data["h0"] = parameter_values

            if "global_tail" in handle:
                tail = handle["global_tail"]
                row["tail_percentile"] = float(
                    tail.attrs.get("probability", math.nan)
                )
                tail_values: dict[str, Any] = {}
                for output_name, dataset_name in (
                    ("m1_scale", "m1_scale"),
                    ("m2_scale", "m2_scale"),
                    (
                        "both_fraction_at_m1_scale",
                        "both_fraction_at_m1_scale",
                    ),
                    (
                        "straddle_fraction_at_m1_scale",
                        "straddle_fraction_at_m1_scale",
                    ),
                ):
                    if dataset_name in tail:
                        values = np.asarray(tail[dataset_name], dtype=float)
                        tail_values[output_name] = values
                        interval = _weighted_quantiles(values, weights)
                        for suffix, value in zip(
                            ("lower", "median", "upper"), interval
                        ):
                            row[f"{output_name}_{suffix}"] = value
                if "m1_scale" in tail_values and "m2_scale" in tail_values:
                    m1_scale = tail_values["m1_scale"]
                    m2_scale = tail_values["m2_scale"]
                    derived = {
                        "m1_minus_m2": m1_scale - m2_scale,
                        "m1_over_m2": np.divide(
                            m1_scale,
                            m2_scale,
                            out=np.full_like(m1_scale, np.nan),
                            where=m2_scale > 0.0,
                        ),
                    }
                    for output_name, values in derived.items():
                        interval = _weighted_quantiles(values, weights)
                        for suffix, value in zip(
                            ("lower", "median", "upper"), interval
                        ):
                            row[f"{output_name}_{suffix}"] = value
    except OSError as exc:
        raise BatchError(f"Cannot read {run.output_directory / 'results.h5'}: {exc}") from exc

    tail_rows = _read_csv_rows(run.output_directory / "global_tail_scales.csv")
    if tail_rows:
        tail = tail_rows[0]
        mapping = {
            "tail_percentile": "percentile",
            "m1_scale_lower": "m1_scale_lower",
            "m1_scale_median": "m1_scale_median",
            "m1_scale_upper": "m1_scale_upper",
            "m2_scale_lower": "m2_scale_lower",
            "m2_scale_median": "m2_scale_median",
            "m2_scale_upper": "m2_scale_upper",
            "m1_minus_m2_lower": "m1_minus_m2_lower",
            "m1_minus_m2_median": "m1_minus_m2_median",
            "m1_minus_m2_upper": "m1_minus_m2_upper",
            "m1_over_m2_lower": "m1_over_m2_lower",
            "m1_over_m2_median": "m1_over_m2_median",
            "m1_over_m2_upper": "m1_over_m2_upper",
            "both_fraction_at_m1_scale_lower": "both_fraction_at_m1_scale_lower",
            "both_fraction_at_m1_scale_median": "both_fraction_at_m1_scale_median",
            "both_fraction_at_m1_scale_upper": "both_fraction_at_m1_scale_upper",
            "straddle_fraction_at_m1_scale_lower": "straddle_fraction_at_m1_scale_lower",
            "straddle_fraction_at_m1_scale_median": "straddle_fraction_at_m1_scale_median",
            "straddle_fraction_at_m1_scale_upper": "straddle_fraction_at_m1_scale_upper",
            "h0_m1_tail_rho": "h0_mu1_rho",
            "h0_m1_tail_mi_bits": "h0_mu1_mi_bits",
            "h0_m2_tail_rho": "h0_mu2_rho",
            "h0_m2_tail_mi_bits": "h0_mu2_mi_bits",
            "h0_tail_vector_mi_bits": "h0_mu_vector_mi_bits",
        }
        for output_name, input_name in mapping.items():
            row[output_name] = _float_value(tail, input_name)

    association_rows = _read_csv_rows(
        run.output_directory / "external_parameter_associations.csv"
    )
    if not association_rows:
        association_rows = _read_csv_rows(
            run.output_directory / "h0_feature_associations.csv"
        )
    tail_association = next(
        (
            item
            for item in association_rows
            if item.get("type", "") == "global_tail_scale"
            and item.get("parameter_name", "") in {"", "H0"}
        ),
        None,
    )
    if tail_association:
        row.update(
            h0_m1_tail_rho=_float_value(tail_association, "mu1_rho"),
            h0_m1_tail_mi_bits=_float_value(
                tail_association, "mu1_mi_bits"
            ),
            h0_m2_tail_rho=_float_value(tail_association, "mu2_rho"),
            h0_m2_tail_mi_bits=_float_value(
                tail_association, "mu2_mi_bits"
            ),
            h0_tail_vector_mi_bits=_float_value(
                tail_association, "mu_vector_mi_bits"
            ),
        )
    return row, plot_data


def _unique_display_labels(runs: list[AggregationRun]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.display_label] = counts.get(run.display_label, 0) + 1
    return {
        run.run_directory.name: (
            run.display_label
            if counts[run.display_label] == 1
            else f"{run.display_label} [{run.run_directory.name}]"
        )
        for run in runs
    }


def _save_figure(fig: Any, stem: Path, comparison: ComparisonConfig) -> None:
    if comparison.write_pdf:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if comparison.write_png:
        fig.savefig(
            stem.with_suffix(".png"),
            dpi=comparison.figure_dpi,
            bbox_inches="tight",
        )


def _matrix_status_suffix(status: str, relation: str = "") -> str:
    suffix = "?" if status == "ambiguous" else ""
    if relation == "split":
        suffix += "S"
    elif relation == "merge":
        suffix += "M"
    elif relation == "split_merge":
        suffix += "SM"
    return suffix


def _plot_baseline_feature_robustness(
    runs: list[AggregationRun],
    feature_rows: list[dict[str, Any]],
    baseline_features: list[dict[str, Any]],
    output_directory: Path,
    comparison: ComparisonConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise BatchError("matplotlib and numpy are required to make figures.") from exc

    usable_runs = [run for run in runs if _completed_aggregation_run(run)]
    labels = _unique_display_labels(usable_runs)
    if not baseline_features:
        return
    lookup = {(row["run"], row["baseline_ID"]): row for row in feature_rows}
    n_runs = len(usable_runs)
    n_features = len(baseline_features)
    p_morph = np.full((n_runs, n_features), np.nan)
    displacement = np.full((n_runs, n_features), np.nan)
    status = np.full((n_runs, n_features), "", dtype=object)
    relation = np.full((n_runs, n_features), "", dtype=object)
    for i, run in enumerate(usable_runs):
        for j, feature in enumerate(baseline_features):
            row = lookup[(run.run_directory.name, feature["ID"])]
            p_morph[i, j] = _float_value(row, "P_morph")
            displacement[i, j] = _float_value(
                row, "centroid_approx_percent_displacement"
            )
            status[i, j] = row.get("match_status", "")
            relation[i, j] = row.get("structural_relation", "")

    figure_height = max(5.8, 0.48 * n_runs + 1.8)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(13.0, 0.80 * n_features + 5.0), figure_height),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    masked_p = np.ma.masked_invalid(p_morph)
    image_p = axes[0].imshow(masked_p, vmin=0.0, vmax=1.0, cmap="Purples", aspect="auto")
    finite_displacement = displacement[np.isfinite(displacement)]
    vmax = (
        max(10.0, float(np.quantile(finite_displacement, 0.95)))
        if finite_displacement.size
        else 10.0
    )
    image_d = axes[1].imshow(
        np.ma.masked_invalid(displacement),
        vmin=0.0,
        vmax=vmax,
        cmap="YlOrRd",
        aspect="auto",
    )

    for axis in axes:
        axis.set_xticks(range(n_features))
        axis.set_xticklabels(
            [f"{feature['feature_type']}\n{feature['ID']}" for feature in baseline_features],
            fontsize=8.5,
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(n_runs))
        axis.set_yticklabels(
            [labels[run.run_directory.name] for run in usable_runs], fontsize=10
        )
        axis.tick_params(length=0)
        axis.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, n_runs, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)

    axes[0].set_title(r"Morphology confidence $P_{\rm morph}$", fontsize=13)
    axes[1].set_title(r"Feature displacement from baseline", fontsize=13)
    for i in range(n_runs):
        for j in range(n_features):
            marker = _matrix_status_suffix(status[i, j], relation[i, j])
            if np.isfinite(p_morph[i, j]):
                colour = "white" if p_morph[i, j] > 0.58 else "black"
                axes[0].text(
                    j, i, f"{p_morph[i, j]:.2f}{marker}",
                    ha="center", va="center", fontsize=8.5, color=colour,
                )
            else:
                text = (
                    "FAIL"
                    if status[i, j] == "run_failed"
                    else f"ND{_matrix_status_suffix(status[i, j], relation[i, j])}"
                )
                axes[0].text(j, i, text, ha="center", va="center", fontsize=8, color="0.35")
            if np.isfinite(displacement[i, j]):
                colour = "white" if displacement[i, j] > 0.58 * vmax else "black"
                axes[1].text(
                    j, i, f"{displacement[i, j]:.0f}%{marker}",
                    ha="center", va="center", fontsize=8.5, color=colour,
                )
            else:
                text = (
                    "FAIL"
                    if status[i, j] == "run_failed"
                    else f"ND{_matrix_status_suffix(status[i, j], relation[i, j])}"
                )
                axes[1].text(j, i, text, ha="center", va="center", fontsize=8, color="0.35")

    fig.colorbar(image_p, ax=axes[0], fraction=0.035, pad=0.025)
    colourbar = fig.colorbar(image_d, ax=axes[1], fraction=0.035, pad=0.025)
    colourbar.set_label(r"$100\,\|\Delta\ln\boldsymbol{\mu}\|$", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "ND: no compatible catalogue feature; ?: ambiguous cross-match; "
        "S/M: compatible split/merge relation. "
        r"$P_{\rm reg}$ is intentionally not used as feature confidence.",
        ha="center",
        fontsize=9.5,
        color="0.3",
    )
    fig.subplots_adjust(bottom=0.07)
    _save_figure(fig, output_directory / "feature_robustness_matrix", comparison)
    plt.close(fig)


def _plot_feature_family_robustness(
    runs: list[AggregationRun],
    family_rows: list[dict[str, Any]],
    family_matrix_rows: list[dict[str, Any]],
    output_directory: Path,
    comparison: ComparisonConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise BatchError("matplotlib and numpy are required to make figures.") from exc

    usable_runs = [run for run in runs if _completed_aggregation_run(run)]
    if not usable_runs or not family_rows:
        return
    labels = _unique_display_labels(usable_runs)
    family_ids = [str(row["family_ID"]) for row in family_rows]
    lookup = {
        (str(row["run"]), str(row["family_ID"])): row
        for row in family_matrix_rows
    }
    n_runs, n_features = len(usable_runs), len(family_rows)
    p_morph = np.full((n_runs, n_features), np.nan)
    displacement = np.full((n_runs, n_features), np.nan)
    status = np.full((n_runs, n_features), "", dtype=object)
    for i, run in enumerate(usable_runs):
        for j, family_id in enumerate(family_ids):
            row = lookup[(run.run_directory.name, family_id)]
            p_morph[i, j] = _float_value(row, "P_morph")
            distance = _float_value(row, "representative_distance_log")
            displacement[i, j] = 100.0 * distance if math.isfinite(distance) else math.nan
            status[i, j] = row.get("family_status", "")

    figure_height = max(5.8, 0.48 * n_runs + 1.8)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(13.0, 0.80 * n_features + 5.0), figure_height),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    image_p = axes[0].imshow(
        np.ma.masked_invalid(p_morph),
        vmin=0.0,
        vmax=1.0,
        cmap="Purples",
        aspect="auto",
    )
    finite = displacement[np.isfinite(displacement)]
    vmax = max(10.0, float(np.quantile(finite, 0.95))) if finite.size else 10.0
    image_d = axes[1].imshow(
        np.ma.masked_invalid(displacement),
        vmin=0.0,
        vmax=vmax,
        cmap="YlOrRd",
        aspect="auto",
    )
    family_labels = [
        f"{row['feature_type']}\n{row['family_ID']}" for row in family_rows
    ]
    for axis in axes:
        axis.set_xticks(range(n_features))
        axis.set_xticklabels(family_labels, fontsize=8.5, rotation=35, ha="right")
        axis.set_yticks(range(n_runs))
        axis.set_yticklabels(
            [labels[run.run_directory.name] for run in usable_runs], fontsize=10
        )
        axis.tick_params(length=0)
        axis.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, n_runs, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_title(r"Reference-free family $P_{\rm morph}$", fontsize=13)
    axes[1].set_title("Displacement from family medoid", fontsize=13)
    for i in range(n_runs):
        for j in range(n_features):
            if np.isfinite(p_morph[i, j]):
                colour = "white" if p_morph[i, j] > 0.58 else "black"
                marker = "S" if status[i, j] == "split" else ""
                axes[0].text(
                    j,
                    i,
                    f"{p_morph[i, j]:.2f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=colour,
                )
            else:
                text = "FAIL" if status[i, j] == "run_failed" else "ND"
                axes[0].text(j, i, text, ha="center", va="center", fontsize=8, color="0.35")
            if np.isfinite(displacement[i, j]):
                colour = "white" if displacement[i, j] > 0.58 * vmax else "black"
                axes[1].text(
                    j,
                    i,
                    f"{displacement[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=colour,
                )
            else:
                text = "FAIL" if status[i, j] == "run_failed" else "ND"
                axes[1].text(j, i, text, ha="center", va="center", fontsize=8, color="0.35")
    fig.colorbar(image_p, ax=axes[0], fraction=0.035, pad=0.025)
    colourbar = fig.colorbar(image_d, ax=axes[1], fraction=0.035, pad=0.025)
    colourbar.set_label(r"$100\,d_{\rm family}$", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "ND: family not detected in that run; S: multiple run features in one family.",
        ha="center",
        fontsize=9.5,
        color="0.3",
    )
    fig.subplots_adjust(bottom=0.14)
    _save_figure(fig, output_directory / "feature_family_robustness_matrix", comparison)
    plt.close(fig)


_ROBUSTNESS_ALT_COLOURS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def _robustness_run_colours(
    runs: list[AggregationRun], baseline: AggregationRun
) -> dict[str, str]:
    colours = {baseline.run_directory.name: "black"}
    index = 0
    for run in runs:
        name = run.run_directory.name
        if name == baseline.run_directory.name:
            continue
        colours[name] = _ROBUSTNESS_ALT_COLOURS[
            index % len(_ROBUSTNESS_ALT_COLOURS)
        ]
        if re.search(r"(?:^|[_-])R4(?:$|[_-])", name):
            colours[name] = "#CC3311"
        index += 1
    return colours


def _weighted_kde_1d(values: Any, weights: Any, grid: Any) -> Any:
    import numpy as np

    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return np.full_like(grid, np.nan, dtype=float)
    values = values[usable]
    weights = weights[usable]
    weights = weights / weights.sum()
    mean = float(np.sum(weights * values))
    variance = float(np.sum(weights * (values - mean) ** 2))
    effective_n = float(1.0 / np.sum(weights**2))
    bandwidth = 1.06 * math.sqrt(max(variance, 0.0)) * effective_n ** (-0.2)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        bandwidth = 1.0
    density = np.zeros_like(grid, dtype=float)
    normalization = bandwidth * math.sqrt(2.0 * math.pi)
    for start in range(0, values.size, 2048):
        stop = min(values.size, start + 2048)
        z = (grid[:, None] - values[None, start:stop]) / bandwidth
        density += np.exp(-0.5 * z**2) @ weights[start:stop]
    return density / normalization


def _plot_h0_posterior_robustness(
    runs: list[AggregationRun],
    plot_data: dict[str, dict[str, Any]],
    baseline: AggregationRun,
    output_directory: Path,
    comparison: ComparisonConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise BatchError("matplotlib and numpy are required to make figures.") from exc

    stem = output_directory / "h0_posterior_robustness"
    for suffix in (".pdf", ".png"):
        previous = stem.with_suffix(suffix)
        if previous.is_file():
            previous.unlink()

    usable_runs = [
        run
        for run in runs
        if run.run_directory.name in plot_data
        and "h0" in plot_data[run.run_directory.name]
    ]
    if not usable_runs or baseline.run_directory.name not in {
        run.run_directory.name for run in usable_runs
    }:
        return
    labels = _unique_display_labels(usable_runs)
    colours = _robustness_run_colours(usable_runs, baseline)
    limits = []
    for run in usable_runs:
        data = plot_data[run.run_directory.name]
        limits.extend(
            _weighted_quantiles(
                data["h0"], data["weights"], (0.001, 0.999)
            )
        )
    limits = [value for value in limits if math.isfinite(value)]
    if not limits:
        return
    lower, upper = min(limits), max(limits)
    padding = 0.08 * (upper - lower) if upper > lower else 1.0
    grid = np.linspace(lower - padding, upper + padding, 600)

    fig, axis = plt.subplots(figsize=(6.8, 4.3))
    ordered = [
        run for run in usable_runs
        if run.run_directory.name != baseline.run_directory.name
    ] + [baseline]
    for run in ordered:
        name = run.run_directory.name
        data = plot_data[name]
        density = _weighted_kde_1d(data["h0"], data["weights"], grid)
        is_baseline = name == baseline.run_directory.name
        axis.plot(
            grid,
            density,
            color=colours[name],
            lw=2.6 if is_baseline else 1.8,
            alpha=1.0 if is_baseline else 0.95,
            label=labels[name],
            zorder=5 if is_baseline else 3,
        )
        if is_baseline:
            axis.fill_between(
                grid,
                0.0,
                density,
                color="black",
                alpha=0.12,
                linewidth=0,
                zorder=4,
            )
    axis.set_xlim(grid[0], grid[-1])
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(r"$H_0\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$")
    axis.set_ylabel("Posterior density")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=7.5, ncol=1, loc="best")
    fig.tight_layout()
    _save_figure(fig, stem, comparison)
    plt.close(fig)


def _plot_marginal_robustness(
    runs: list[AggregationRun],
    marginals: dict[str, dict[str, Any]],
    baseline: AggregationRun,
    output_directory: Path,
    comparison: ComparisonConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise BatchError("matplotlib and numpy are required to make figures.") from exc

    usable_runs = [run for run in runs if run.run_directory.name in marginals]
    if baseline.run_directory.name not in marginals:
        return
    labels = _unique_display_labels(usable_runs)
    colours = _robustness_run_colours(usable_runs, baseline)
    variants = [
        run
        for run in usable_runs
        if run.run_directory.name != baseline.run_directory.name
    ]
    split = (len(variants) + 1) // 2
    groups = (variants[:split], variants[split:])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.5),
        sharex=True,
        sharey=True,
    )
    baseline_data = marginals[baseline.run_directory.name]
    all_upper = []
    for run in usable_runs:
        quantiles = np.asarray(
            marginals[run.run_directory.name]["marginal1"], dtype=float
        )
        finite = quantiles[2][np.isfinite(quantiles[2]) & (quantiles[2] > 0.0)]
        if finite.size:
            all_upper.append(float(np.max(finite)))
    ymax = 1.6 * max(all_upper) if all_upper else 1.0

    for panel, (axis, group) in enumerate(zip(axes, groups), start=1):
        grid = baseline_data["m1"]
        quantiles = baseline_data["marginal1"]
        axis.fill_between(
            grid,
            quantiles[0],
            quantiles[2],
            color="0.25",
            alpha=0.22,
            linewidth=0,
            label=labels[baseline.run_directory.name],
            zorder=3,
        )
        axis.plot(grid, quantiles[0], color="black", lw=1.7, zorder=4)
        axis.plot(grid, quantiles[2], color="black", lw=1.7, zorder=4)
        for run in group:
            name = run.run_directory.name
            data = marginals[run.run_directory.name]
            colour = colours[name]
            q = data["marginal1"]
            axis.fill_between(
                data["m1"],
                q[0],
                q[2],
                color=colour,
                alpha=0.055,
                linewidth=0,
                label=labels[name],
                zorder=1,
            )
            axis.plot(
                data["m1"],
                q[0],
                color=colour,
                lw=1.25,
                ls="--",
                alpha=0.95,
                zorder=3,
            )
            axis.plot(
                data["m1"],
                q[2],
                color=colour,
                lw=1.25,
                ls="--",
                alpha=0.95,
                zorder=3,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(float(grid[0]), float(grid[-1]))
        axis.set_ylim(1.0e-5, ymax)
        axis.set_xlabel(r"$m_1\,[M_\odot]$", fontsize=12)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=7.5, ncol=1, loc="best")
        axis.set_title(f"Model variants {panel}/2", fontsize=12)
    axes[0].set_ylabel(r"$p_{\ln,1}(m_1)$", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, output_directory / "marginal_reconstruction_robustness", comparison)
    plt.close(fig)


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = [_float_value(row, field) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _write_robustness_summary(
    output_directory: Path,
    runs: list[AggregationRun],
    baseline: AggregationRun,
    feature_rows: list[dict[str, Any]],
    baseline_features: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    additional_rows: list[dict[str, Any]],
    split_merge_rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
) -> None:
    completed = [run for run in runs if _completed_aggregation_run(run)]
    lines = [
        "# Posterior-landscape model-robustness summary",
        "",
        f"Baseline: `{baseline.run_directory.name}` ({baseline.display_label}).",
        f"Completed outputs included: {len(completed)} of {len(runs)} discovered runs.",
        "",
        "The ranges below compare model variants. They are not pooled posterior "
        "intervals and the variants are not assigned model weights. A missing "
        "cross-match is reported as `not detected`, not as $P_{\\rm morph}=0$.",
        "",
        "## Feature stability",
        "",
        "| Baseline feature | Detected runs | Not detected | Ambiguous | Split/merge | "
        "$P_{\\rm morph}$ range | Median centroid range $[M_\\odot]$ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in baseline_features:
        rows = [
            row for row in feature_rows
            if row["baseline_ID"] == feature["ID"]
            and row["run_status"] in {"completed", "completed_existing"}
        ]
        matched = [
            row for row in rows
            if row.get("match_status") in {"baseline", "matched", "ambiguous"}
            and math.isfinite(_float_value(row, "P_morph"))
        ]
        missing = sum(row.get("match_status") == "not_detected" for row in rows)
        ambiguous = sum(row.get("match_status") == "ambiguous" for row in rows)
        relations = sum(
            row.get("structural_relation", "none") != "none" for row in rows
        )
        p_values = _finite_values(matched, "P_morph")
        mu1 = _finite_values(matched, "mu1_median")
        mu2 = _finite_values(matched, "mu2_median")
        p_text = f"{min(p_values):.3f}--{max(p_values):.3f}" if p_values else "--"
        centroid_text = (
            f"$m_1$: {min(mu1):.2f}--{max(mu1):.2f}; "
            f"$m_2$: {min(mu2):.2f}--{max(mu2):.2f}"
            if mu1 and mu2 else "--"
        )
        lines.append(
            f"| {feature['feature_type']} `{feature['ID']}` | {len(matched)}/{len(rows)} | "
            f"{missing} | {ambiguous} | {relations} | {p_text} | {centroid_text} |"
        )

    lines.extend(["", "## Reference-free feature families", ""])
    lines.append(
        f"- Families in the union of completed catalogues: {len(family_rows)}."
    )
    lines.append(
        f"- Families absent from the baseline: {len(additional_rows)}."
    )
    if additional_rows:
        singleton_count = sum(
            _float_value(row, "number_runs_detected") == 1.0
            for row in additional_rows
        )
        lines.append(
            f"- Additional families found in only one completed run: {singleton_count}."
        )
    split_count = sum(row.get("relation_type") == "split" for row in split_merge_rows)
    merge_count = sum(row.get("relation_type") == "merge" for row in split_merge_rows)
    lines.append(
        f"- Compatibility-based split relations: {split_count}; merge relations: {merge_count}."
    )

    lines.extend(["", "## Global posterior and tail scales", ""])
    completed_global = [
        row for row in global_rows
        if row["run_status"] in {"completed", "completed_existing"}
    ]
    for field, label, units in (
        ("h0_median", "$H_0$ median", r"km s$^{-1}$ Mpc$^{-1}$"),
        ("m1_scale_median", "$m_1$ upper-tail median", r"$M_\odot$"),
        ("m2_scale_median", "$m_2$ upper-tail median", r"$M_\odot$"),
        ("straddle_fraction_at_m1_scale_median", "Straddling fraction median", ""),
    ):
        values = _finite_values(completed_global, field)
        if values:
            lines.append(
                f"- {label}: {min(values):.3g}--{max(values):.3g} {units}".rstrip()
                + "."
            )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Use `P_morph` as the posterior confidence in a named morphology.",
            "- Use `P_reg` only to diagnose whether its complete finite geometry "
            "was measurable; it is retained in the CSV but not used in the confidence matrix.",
            "- Boundary status is retained as metadata and is not a comparison failure.",
            "- Split/merge rows are compatibility relations, not silently merged feature identities.",
            "- Inspect rows marked `ambiguous` before making a feature-by-feature claim.",
            "- Do not average confidence values across variants unless an explicit "
            "model-averaging prior is introduced.",
            "",
        ]
    )
    _write_atomic(
        output_directory / "robustness_summary.md",
        "\n".join(lines).encode("utf-8"),
    )


def aggregate_batch_results(config: BatchConfig) -> int:
    comparison = config.comparison
    if not comparison.enabled:
        print("Robustness aggregation is disabled in [comparison].")
        return 0
    runs = collect_aggregation_runs(config)
    if not runs:
        raise BatchError("No runs are available for robustness aggregation.")
    baseline = _select_baseline_run(runs, comparison)
    comparison.output_directory.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    split_merge_rows: list[dict[str, Any]] = []
    marginals: dict[str, dict[str, Any]] = {}
    candidates_by_run: dict[str, list[dict[str, Any]]] = {}

    for index, run in enumerate(runs, start=1):
        print(f"Robustness aggregation: {index}/{len(runs)} ({run.display_label}).")
        if _completed_aggregation_run(run):
            candidates = _standard_feature_candidates(run.output_directory)
            candidates_by_run[run.run_directory.name] = candidates
        global_row, marginal_data = _global_row_for_run(run)
        global_rows.append(global_row)
        if marginal_data is not None:
            marginals[run.run_directory.name] = marginal_data

    baseline_features = candidates_by_run.get(baseline.run_directory.name, [])
    if not baseline_features:
        raise BatchError(
            "The selected baseline has no primary morphology entries in its "
            "catalogue; no baseline-referenced matrix can be constructed."
        )

    for run in runs:
        candidates = candidates_by_run.get(run.run_directory.name, [])
        if not _completed_aggregation_run(run):
            assignments: dict[str, dict[str, Any]] = {}
        elif run.run_directory.name == baseline.run_directory.name:
            assignments = _baseline_assignments(baseline_features)
        else:
            assignments = _assign_baseline_features(
                baseline_features, candidates, comparison
            )
            split_merge_rows.extend(
                _split_merge_rows_for_run(
                    run, baseline_features, candidates, comparison
                )
            )
        feature_rows.extend(
            _baseline_feature_rows_for_run(run, baseline_features, assignments)
        )

    families = _build_feature_families(
        candidates_by_run, runs, baseline, comparison
    )
    family_rows = _family_summary_rows(families, runs, baseline)
    family_matrix_rows = _feature_family_matrix_rows(families, runs, baseline)
    additional_rows = _additional_feature_rows(
        families, baseline_features, runs, baseline
    )

    output = comparison.output_directory
    _write_csv_atomic(
        output / "robustness_features.csv",
        feature_rows,
        BASELINE_FEATURE_FIELDS,
    )
    _write_csv_atomic(
        output / "robustness_global.csv",
        global_rows,
        GLOBAL_SUMMARY_FIELDS,
    )
    _write_csv_atomic(
        output / "robustness_feature_families.csv",
        family_rows,
        FEATURE_FAMILY_FIELDS,
    )
    _write_csv_atomic(
        output / "robustness_feature_family_matrix.csv",
        family_matrix_rows,
        FEATURE_FAMILY_MATRIX_FIELDS,
    )
    _write_csv_atomic(
        output / "robustness_additional_features.csv",
        additional_rows,
        ADDITIONAL_FEATURE_FIELDS,
    )
    _write_csv_atomic(
        output / "robustness_split_merge_relations.csv",
        split_merge_rows,
        SPLIT_MERGE_FIELDS,
    )
    obsolete_relations = output / "robustness_relations.csv"
    if obsolete_relations.is_file():
        obsolete_relations.unlink()
    manifest = {
        "aggregation_schema": "3.0",
        "created_utc": _utc_now(),
        "baseline_run": baseline.run_directory.name,
        "baseline_control_output": str(
            baseline.control_directory or baseline.output_directory
        ),
        "baseline_output": str(baseline.output_directory),
        "number_discovered_runs": len(runs),
        "number_completed_runs": sum(_completed_aggregation_run(run) for run in runs),
        "draws_per_chain_request": config.requested_draws_per_chain,
        "feature_matching": {
            "maximum_log_centroid_distance": comparison.maximum_log_centroid_distance,
            "minimum_match_margin": comparison.minimum_match_margin,
            "not_detected_is_zero_confidence": False,
            "primary_feature_types": list(PRIMARY_FEATURE_TYPES),
            "paired_events": "derived_only",
            "boundary_status": "metadata_not_failure",
            "baseline_matrix": "baseline_catalogue_rows",
            "reference_free_matrix": "all-run compatible feature families",
            "split_merge": "reported_as_compatible_relations",
        },
    }
    _write_atomic(
        output / "robustness_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write_robustness_summary(
        output,
        runs,
        baseline,
        feature_rows,
        baseline_features,
        family_rows,
        additional_rows,
        split_merge_rows,
        global_rows,
    )
    _plot_baseline_feature_robustness(
        runs, feature_rows, baseline_features, output, comparison
    )
    _plot_feature_family_robustness(
        runs, family_rows, family_matrix_rows, output, comparison
    )
    _plot_h0_posterior_robustness(
        runs, marginals, baseline, output, comparison
    )
    _plot_marginal_robustness(
        runs, marginals, baseline, output, comparison
    )
    for stale_stem in (
        "h0_global_scale_robustness",
        "global_tail_scale_robustness",
        "m99_robustness",
    ):
        for suffix in (".pdf", ".png"):
            stale = output / f"{stale_stem}{suffix}"
            if stale.is_file():
                stale.unlink()
    print(f"Robustness products: {output}")
    return 0


def run_batch(config: BatchConfig) -> int:
    run_directories = discover_run_directories(config)
    environment, working_directory = build_execution_environment(
        config.posterior_landscape_root
    )
    print(
        f"Found {len(run_directories)} run directories; "
        f"requested draws/chain={config.requested_draws_per_chain}."
    )
    if config.dry_run:
        print("Dry run: inputs will be validated, but nothing will be written or executed.")

    records: list[dict[str, Any]] = []
    failed = 0
    for index, run_directory in enumerate(run_directories, start=1):
        started_utc = _utc_now()
        started = time.monotonic()
        density: DensityInfo | None = None
        output_directory: Path | None = None
        try:
            density = select_density_file(run_directory, config)
            output_directory, settings_path = prepare_run_files(
                run_directory, density, config
            )
            print(
                f"\n[{index}/{len(run_directories)}] {run_directory.name}\n"
                f"  input:  {density.path.name}\n"
                f"  draws:  {_counts_text(density.actual_draws_per_chain)} "
                f"(total {density.number_draws})\n"
                f"  output: {output_directory}"
            )

            if config.dry_run:
                return_code = 0
                status = "dry_run_ok"
                message = "validated; not executed"
            else:
                complete, completion_message = _workflow_completion_status(
                    output_directory,
                    settings_path,
                )
                if complete:
                    print("  complete output found; skipping posterior-landscape.")
                    return_code = 0
                    status = "completed_existing"
                    message = "complete output reused; analysis was not invoked"
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "posterior_landscape",
                        str(settings_path),
                    ]
                    resumable_products = (
                        ".posterior_landscape.checkpoint.pkl.gz",
                        "one_dimensional_state.pkl.gz",
                        "one_dimensional_m2_state.pkl.gz",
                        "manifest.json",
                        "results.h5",
                    )
                    has_partial_products = any(
                        (output_directory / name).is_file()
                        for name in resumable_products
                    )
                    if has_partial_products:
                        print(
                            "  incomplete output found; invoking posterior-landscape "
                            f"to resume ({completion_message})."
                        )
                    else:
                        print("  running posterior-landscape...")
                    completed = subprocess.run(
                        command,
                        cwd=working_directory,
                        env=environment,
                    )
                    return_code = int(completed.returncode)
                    if return_code == 0:
                        complete, completion_message = _workflow_completion_status(
                            output_directory,
                            settings_path,
                        )
                        if complete:
                            status = "completed"
                            message = ""
                        else:
                            status = "failed"
                            message = (
                                "posterior-landscape returned success but the "
                                f"requested workflow is incomplete ({completion_message})"
                            )
                            failed += 1
                    else:
                        status = "failed"
                        message = f"posterior-landscape exited with code {return_code}"
                        failed += 1
        except KeyboardInterrupt:
            record = {
                "run": run_directory.name,
                "status": "interrupted",
                "return_code": 130,
                "requested_draws_per_chain": config.requested_draws_per_chain,
                "actual_draws_per_chain": (
                    _counts_text(density.actual_draws_per_chain) if density else ""
                ),
                "number_draws": density.number_draws if density else "",
                "input_file": str(density.path) if density else "",
                "output_directory": str(output_directory) if output_directory else "",
                "started_utc": started_utc,
                "finished_utc": _utc_now(),
                "elapsed_seconds": f"{time.monotonic() - started:.3f}",
                "message": "interrupted by user",
            }
            records.append(record)
            if not config.dry_run:
                write_status_csv(config.status_csv, records)
            print("\nInterrupted; rerun the same command to resume.")
            return 130
        except Exception as exc:
            return_code = 1
            status = "error"
            message = str(exc)
            failed += 1
            print(f"\n[{index}/{len(run_directories)}] {run_directory.name}: ERROR: {message}")

        record = {
            "run": run_directory.name,
            "status": status,
            "return_code": return_code,
            "requested_draws_per_chain": config.requested_draws_per_chain,
            "actual_draws_per_chain": (
                _counts_text(density.actual_draws_per_chain) if density else ""
            ),
            "number_draws": density.number_draws if density else "",
            "input_file": str(density.path) if density else "",
            "output_directory": str(output_directory) if output_directory else "",
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            "elapsed_seconds": f"{time.monotonic() - started:.3f}",
            "message": message,
        }
        records.append(record)
        if not config.dry_run:
            write_status_csv(config.status_csv, records)

        if return_code != 0 and not config.continue_on_error:
            print("Stopping because continue_on_error=false.")
            break

    completed_count = sum(
        record["status"] in {"completed", "completed_existing"}
        for record in records
    )
    reused_count = sum(
        record["status"] == "completed_existing" for record in records
    )
    dry_count = sum(record["status"] == "dry_run_ok" for record in records)
    print(
        f"\nBatch finished: complete={completed_count} (reused={reused_count}), "
        f"dry-run={dry_count}, "
        f"failed={failed}."
    )
    if not config.dry_run:
        print(f"Status table: {config.status_csv}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run posterior-landscape over cached diagnostic-run densities."
    )
    parser.add_argument(
        "settings",
        nargs="?",
        default="batch_settings.ini",
        help="batch INI file (default: batch_settings.ini)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "read existing per-run outputs and regenerate robustness tables and "
            "figures without invoking posterior-landscape"
        ),
    )
    mode.add_argument(
        "--run-only",
        action="store_true",
        help="run posterior-landscape but skip the final robustness aggregation",
    )
    arguments = parser.parse_args(argv)
    try:
        config = load_batch_config(Path(arguments.settings))
        if arguments.aggregate_only:
            return aggregate_batch_results(config)
        batch_code = run_batch(config)
        if (
            not arguments.run_only
            and not config.dry_run
            and config.comparison.enabled
        ):
            aggregate_code = aggregate_batch_results(config)
            return batch_code if batch_code else aggregate_code
        return batch_code
    except BatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
