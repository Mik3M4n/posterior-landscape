#!/usr/bin/env python3
"""Compare legacy and corrected reference catalogues from a saved results.h5.

This audit is read-only and needs no original posterior draws. It is a
reference-field acceptance check, not a replacement for a posterior rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posterior_landscape import __version__
from posterior_landscape.config import load_settings
from posterior_landscape.ensemble import (
    add_scale_persistence, assign_reference_identifiers,
    remove_corner_boundary_pits, retain_scale_persistent_reference,
)
from posterior_landscape.io import Grid, diagonal_cells
from posterior_landscape.quadrature import build_axis_quadrature
from posterior_landscape.topology import analyze_field, hpd_threshold


def audit(results: Path, settings_path: Path) -> dict:
    settings = load_settings(settings_path)
    with h5py.File(results, "r") as handle:
        attrs = dict(handle.attrs)
        c = handle["coordinates"]
        density = handle["maps/reference_density"][:]
        mass = [c["m1"][:], c["m2"][:]]
        location = attrs.get("grid_location", "nodes")
        native = attrs.get("grid_measure", "linear")
        base = float(attrs["log_base"])
        axes = [build_axis_quadrature(m, str(location), str(native), base) for m in mass]
        geometry = str(attrs["analysis_geometry"])
        ell = [np.log(m)/np.log(base) for m in mass]
        grid = Grid(
            m1=mass[0], m2=mass[1], ell1=ell[0], ell2=ell[1],
            geometry1=ell[0] if geometry=="log" else mass[0],
            geometry2=ell[1] if geometry=="log" else mass[1],
            mask=c["mask"][:].astype(bool), weight1=c["quadrature_m1"][:],
            weight2=c["quadrature_m2"][:], log_base=base, geometry=geometry,
            feature_measure=str(attrs["feature_measure"]),
            input_density_measure=str(attrs["input_density_measure"]),
            domain=str(attrs.get("domain", settings.input.domain)),
            ordered_domain=bool(attrs.get("ordered_domain", settings.input.domain=="ordered")),
            grid_location=str(location), grid_measure=str(native), axis1=axes[0],axis2=axes[1],
        )
        saved = {
            fid: group["reference_curve_geometry"][:]
            for fid, group in handle["features"].items()
            if group.attrs["type"]=="ridge"
        }
        source_mode = (
            str(attrs.get("matching_policy", "legacy")),
            str(attrs.get("ridge_region_policy", "legacy_two_sided")),
        )
        input_integrals = handle["posterior/input_normalization"][:]
    density_before=density.copy(); weights_before=grid.cell_weights.copy()
    output = {
        "package_version":__version__, "results":str(results.resolve()),
        "source_package_version":str(attrs.get("package_version", "unknown")),
        "settings":str(settings_path.resolve()),
        "source_sha256":hashlib.sha256(results.read_bytes()).hexdigest(),
        "scope":"reference fields only; original posterior draws must be rerun separately",
        "grid_location":grid.grid_location, "grid_measure":grid.grid_measure,
        "active_mask_cells":int(grid.mask.sum()),
        "active_diagonal_cells":int(np.count_nonzero(grid.mask & diagonal_cells(*mass))),
        "stored_input_integrals":{
            "draws":int(input_integrals.size),
            "minimum":float(np.min(input_integrals)),
            "q05":float(np.quantile(input_integrals,.05)),
            "median":float(np.median(input_integrals)),
            "q95":float(np.quantile(input_integrals,.95)),
            "maximum":float(np.max(input_integrals)),
            "maximum_absolute_error":float(np.max(np.abs(input_integrals-1))),
            "note":"Stored diagnostics; original draw fields are not reintegrated by this audit",
        },
        "reference_integral":grid.integrate(density),
        "density_hpd_thresholds":{str(p):hpd_threshold(density,p,grid) for p in (.5,.9)},
        "modes":{},
    }
    modes=[("legacy","legacy_two_sided"),("coherent","legacy_two_sided"),("coherent","mask_supported")]
    if source_mode not in modes:
        modes.append(source_mode)
    for matching,region in modes:
        a=settings.analysis
        analyses=[analyze_field(density,grid,scale=scale,
            persistence_threshold=a.persistence_threshold,persistence_gap_min_log=a.persistence_gap_min_log,
            hessian_refinement=a.hessian_refinement,detect_plateaus=False,
            curve_points=a.curve_points,support_mass=a.support_mass,
            curve_smoothing_cells=a.curve_smoothing_cells,detect_shoulders=False,
            ridge_region_policy=region) for scale in a.scales]
        reference=analyses[0]
        diagnostic=add_scale_persistence(reference,analyses[1:],grid,matching_policy=matching)
        scale_filter=retain_scale_persistent_reference(reference)
        corner_filter=remove_corner_boundary_pits(reference)
        assign_reference_identifiers(reference,grid,matching_policy=matching)
        peaks={p.identifier:[float(grid.m1[p.index[0]]),float(grid.m2[p.index[1]])]
               for p in reference.points if p.kind=="peak"}
        arms={}
        for b in reference.branches:
            if b.kind!="ridge":continue
            if b.feature_region_mask is not None:
                if np.any(b.feature_region_mask & ~grid.mask):raise AssertionError("Region escaped mask")
                np.testing.assert_allclose(
                    grid.integrate(np.where(b.feature_region_mask, reference.density, 0)),
                    b.feature_probability, rtol=1e-10,
                )
            arms[b.identifier]={
                "saddle":b.start_identifier,"peak":b.end_identifier,
                "start":b.points_mass[0].tolist(),"end":b.points_mass[-1].tolist(),
                "scale_count":int(b.scale_count),"scale_total":int(b.scale_total),
                "geometry_valid":bool(b.geometry_valid),
                "valid_width_fraction":float(b.valid_width_fraction),
                "boundary_bounded_fraction":float(b.boundary_bounded_fraction),
                "width_median":float(b.width_median) if np.isfinite(b.width_median) else None,
                "probability":float(b.feature_probability) if np.isfinite(b.feature_probability) else None,
            }
        events={e.identifier:{"arms":list(e.arm_identifiers),"region_valid":bool(e.region_valid),
                "probability":float(e.region_probability) if np.isfinite(e.region_probability) else None,
                "centroid":e.mass_centroid.tolist() if np.all(np.isfinite(e.mass_centroid)) else None,
                "boundary_bounded_fraction":float(e.boundary_bounded_fraction)}
                for e in reference.events if e.kind=="ridge"}
        for event in reference.events:
            if event.kind!="ridge" or not event.region_valid:
                continue
            if np.any(event.window_mask & ~grid.mask):raise AssertionError("Event window escaped mask")
            projection_errors={}
            for label, projection, weights in (
                ("m1",event.projection_m1,grid.weight1),
                ("m2",event.projection_m2,grid.weight2),
            ):
                integral=float(projection @ weights)
                # Existing exported projections use float32. Check their own
                # representation precision, and independently check the full
                # float64 marginalization below at numerical precision.
                precision=np.finfo(projection.dtype).eps
                np.testing.assert_allclose(integral,event.region_probability,rtol=8*precision)
                projection_errors[label]=abs(integral-event.region_probability)
            events[event.identifier]["stored_projection_absolute_errors"]=projection_errors
            marginals=grid.marginals(np.where(event.window_mask,reference.density,0))
            np.testing.assert_allclose(marginals[0] @ grid.weight1,event.region_probability,rtol=1e-12)
            np.testing.assert_allclose(marginals[1] @ grid.weight2,event.region_probability,rtol=1e-12)
        mode={"persistence_threshold":float(reference.persistence_threshold),"peaks":peaks,
              "ridge_arms":arms,"paired_ridges":events,"scale_filter":scale_filter,
              "corner_filter":corner_filter,"matching_diagnostics":diagnostic}
        if (matching,region)==source_mode:
            differences=[]
            for fid,curve in saved.items():
                branch=next(b for b in reference.branches if b.identifier==fid)
                np.testing.assert_allclose(branch.points_geometry,curve,rtol=1e-10,atol=1e-10)
                differences.append(float(np.max(np.abs(branch.points_geometry-curve))))
            mode["maximum_saved_curve_difference"]=max(differences,default=0)
        output["modes"][matching+"/"+region]=mode
    np.testing.assert_array_equal(density,density_before)
    np.testing.assert_array_equal(grid.cell_weights,weights_before)
    modes=output["modes"]
    np.testing.assert_equal(modes["legacy/legacy_two_sided"]["peaks"],modes["coherent/mask_supported"]["peaks"])
    if hashlib.sha256(results.read_bytes()).hexdigest()!=output["source_sha256"]:raise AssertionError("Source modified")
    output["source_and_quadrature_unchanged"]=True
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path,help="Original baseline results.h5")
    parser.add_argument("--settings",type=Path,help="Defaults to adjacent settings.ini")
    parser.add_argument("--output",type=Path,help="Optional JSON report")
    parser.add_argument("--expected-peaks",type=int,help="Optional corrected reference peak count")
    parser.add_argument("--expected-arms",type=int,help="Optional corrected reference ridge-arm count")
    parser.add_argument("--expected-events",type=int,help="Optional corrected paired-ridge count")
    args=parser.parse_args()
    report=audit(args.results,args.settings or args.results.parent/"settings.ini")
    corrected=report["modes"]["coherent/mask_supported"]
    for expected,key in ((args.expected_peaks,"peaks"),(args.expected_arms,"ridge_arms"),(args.expected_events,"paired_ridges")):
        if expected is not None and len(corrected[key])!=expected:
            raise AssertionError(f"Expected {expected} {key}, got {len(corrected[key])}")
    text=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding="utf-8")
    print(text,end="")


if __name__=="__main__":main()
