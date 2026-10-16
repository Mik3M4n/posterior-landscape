"""Policy provenance and executable boundary-aware pipeline regressions."""
from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from posterior_landscape import __version__
from posterior_landscape.boundary import RIDGE_REGION_SCHEMA_VERSION
from posterior_landscape.config import load_settings, save_settings
from posterior_landscape.io import h5py
from posterior_landscape.matching import MATCHING_SCHEMA_VERSION
from posterior_landscape.pipeline import (
    _boundary_policy_compatibility, _boundary_policy_metadata,
    _completed_boundary_policy_compatibility, _fingerprint,
    _load_checkpoint, _save_checkpoint, run,
)


def fixture(directory: Path) -> Path:
    mass = np.linspace(1.0, 4.0, 31)
    x, y = np.meshgrid(mass, mass, indexing="ij")
    mask = y <= x
    field = (np.exp(-((x-1)**2+(y-1)**2)/(.2**2*2))
             + .6*np.exp(-((x-2.8)**2+(y-2.8)**2)/(.3**2*2)) + 1e-6)
    w = np.full(mass.size, mass[1]-mass[0]); w[[0,-1]] *= .5
    field = np.where(mask, field, 0)
    field /= np.sum(field*np.multiply.outer(w,w))
    np.savez(directory/"posterior.npz", p=np.stack([field]*3), m1=mass, m2=mass, mask=mask)
    path = directory/"settings.ini"
    path.write_text("""[input]
file = posterior.npz
domain = ordered
density_measure = linear
[analysis]
run = 2d
geometry = linear
feature_measure = linear
scales = 0
persistence_threshold = 0.01
hessian_refinement = false
detect_plateaus = false
detect_shoulders = false
ordered_tails = false
curve_points = 16
curve_smoothing_cells = 0
[compute]
workers = 1
batch_size = 3
resume = true
[association]
enabled = off
[output]
directory = result
profile = full
[plot]
write_pdf = false
write_png = false
""", encoding="utf-8")
    return path


class BoundaryPolicyTests(unittest.TestCase):
    def test_resolved_policies_serialize_and_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary); settings=load_settings(fixture(directory))
            self.assertEqual(settings.analysis.matching_policy, "coherent")
            self.assertEqual(settings.analysis.ridge_region_policy, "mask_supported")
            variants=[replace(settings, analysis=replace(settings.analysis, **change)) for change in
                      [{"matching_policy":"legacy"}, {"ridge_region_policy":"legacy_two_sided"}]]
            hashes={_fingerprint(s,(3,31,31)) for s in [settings,*variants]}
            self.assertEqual(len(hashes),3)
            legacy=replace(settings, analysis=replace(settings.analysis,
                matching_policy="legacy",ridge_region_policy="legacy_two_sided"))
            saved=directory/"saved.ini";save_settings(legacy,saved)
            loaded=load_settings(saved)
            self.assertEqual(loaded.analysis.matching_policy,"legacy")
            self.assertEqual(loaded.analysis.ridge_region_policy,"legacy_two_sided")
            self.assertFalse(_boundary_policy_compatibility({},settings)[0])
            self.assertTrue(_boundary_policy_compatibility({},legacy)[0])
            self.assertTrue(_boundary_policy_compatibility(_boundary_policy_metadata(settings),settings)[0])

    def test_invalid_policy_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=fixture(Path(temporary)); original=path.read_text()
            for name in ("matching_policy", "ridge_region_policy"):
                path.write_text(original.replace("[analysis]",f"[analysis]\n{name}=invalid"))
                with self.assertRaisesRegex(ValueError,name):load_settings(path)

    def test_checkpoint_requires_current_schema_and_requested_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"checkpoint.pkl.gz"
            catalogue=SimpleNamespace(matching_policy="coherent")
            _save_checkpoint(path,fingerprint="same",next_draw=1,catalogue=catalogue)
            self.assertEqual(_load_checkpoint(path,"same",matching_policy="coherent",ridge_region_policy="mask_supported")[0],1)
            with self.assertRaisesRegex(ValueError,"conventions"):
                _load_checkpoint(path,"same",ridge_region_policy="legacy_two_sided")
            with gzip.open(path,"wb") as handle:
                pickle.dump({"version":__version__,"fingerprint":"same","next_draw":1,"catalogue":catalogue},handle)
            with self.assertRaisesRegex(ValueError,"conventions"):_load_checkpoint(path,"same")

    @unittest.skipIf(h5py is None,"h5py is required")
    def test_completed_metadata_checked_in_both_manifest_and_hdf5(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);settings=load_settings(fixture(directory))
            metadata=_boundary_policy_metadata(settings)
            with h5py.File(directory/"results.h5","w") as handle:pass
            self.assertFalse(_completed_boundary_policy_compatibility(directory,metadata,settings)[0])
            with h5py.File(directory/"results.h5","a") as handle:
                for k,v in metadata.items():handle.attrs[k]=v
            self.assertTrue(_completed_boundary_policy_compatibility(directory,metadata,settings)[0])
            self.assertFalse(_completed_boundary_policy_compatibility(directory,{},settings)[0])

    @unittest.skipIf(h5py is None,"h5py is required")
    def test_diagonal_pipeline_keeps_low_endpoint_peak_and_coherent_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);settings=load_settings(fixture(directory))
            output=run(settings)/"full"
            manifest=json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["matching_schema_version"],MATCHING_SCHEMA_VERSION)
            self.assertEqual(manifest["ridge_region_schema_version"],RIDGE_REGION_SCHEMA_VERSION)
            with h5py.File(output/"results.h5","r") as handle:
                features=handle["features"];peaks=[g for g in features.values() if g.attrs["type"]=="peak"]
                self.assertTrue(any(np.allclose(g["location_geometry"][:],[1,1]) for g in peaks))
                self.assertGreaterEqual(len(handle["events"]),1)
                for g in features.values():
                    if g.attrs["type"]!="ridge":continue
                    self.assertTrue(np.all(g["geometry_valid"][:]))
                    self.assertTrue(np.all(g["present"][:]<=features[g.attrs["end_feature"]]["present"][:]))
                    self.assertEqual(g.attrs["region_policy"],"mask_supported")
                    self.assertIn("reference_section_left_boundary_bounded",g)
                    region=g["reference_feature_region_mask"][:]
                    self.assertFalse(np.any(region & ~handle["coordinates/mask"][:].astype(bool)))
                for g in handle["events"].values():
                    a,b=g.attrs["arms"].split("+")
                    np.testing.assert_array_equal(g["present"][:],features[a]["present"][:]&features[b]["present"][:])
                    self.assertTrue(np.all(g["region_valid"][:]))
            self.assertEqual(run(settings),output.parent)
            manifest.pop("matching_schema_version")
            (output/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FileExistsError,"matching_schema_version"):
                run(settings)
