import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

import figure
import pilot
import verify


class TransportTests(unittest.TestCase):
    def test_rigid_motion_rotates_offsets_once(self):
        rest, faces = pilot.grid(5)
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        moved = rest @ rotation.T + np.array([2., 5., -1.])
        inverse_basis = np.linalg.inv(pilot.tri_basis(rest, faces))
        expected = np.broadcast_to(rotation, (len(faces), 3, 3))
        for exponent in (0, 0.5, 1):
            with self.subTest(exponent=exponent):
                actual = pilot.transport(moved, faces, inverse_basis, exponent)
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_custom_run_uses_its_own_peak_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = pilot.read_config(pilot.ROOT / "config.json")
            config.update(assets=["fluted_panel"], policies={"custom": 0.75, "rotation": 0},
                          fixed_strip_max_x=0.27, peak_sequence=[0, 0.8, 0])
            path = folder / "study.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            command = [sys.executable, str(pilot.ROOT / "pilot.py"), "--config", str(path),
                       "--output", "experiment"]
            subprocess.run(command, cwd=folder, check=True, capture_output=True)
            output = folder / "experiment"
            with (output / "measures.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(float(row["protected_boundary_gap_max"]) > 0 for row in rows if row["step"] == "1"))
            peak = figure.read_peak(output, config)
            self.assertGreater(peak["fluted_panel", "custom"], peak["fluted_panel", "rotation"])
            figure.draw_figure(output, output / "transport_check.png")
            with Image.open(output / "cycle_demo.gif") as animation:
                self.assertEqual((animation.n_frames, animation.size), (3, (680, 290)))
            self.assertTrue((output / "environment.json").is_file())
            self.assertNotEqual(subprocess.run(command, cwd=folder, capture_output=True).returncode, 0)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                verify.verify(output)

    def test_verifier_rejects_changed_or_duplicate_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "run"
            shutil.copytree(verify.REFERENCE, copy)
            self.assertEqual(verify.verify(copy), (3, 99, 6))
            path = copy / "measures.csv"
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            rows[1][rows[0].index("protected_max")] = "0.25"
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaisesRegex(ValueError, "protected_max"):
                verify.verify(copy)
            with path.open("a", newline="") as stream:
                csv.writer(stream).writerow(rows[1])
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                verify.verify(copy)

    def test_unknown_field_and_unsupported_right_strip_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unknown relief"):
            pilot.relief(np.zeros((1, 3)), "unlisted_field")
        with tempfile.TemporaryDirectory() as temp:
            config = pilot.read_config(pilot.ROOT / "config.json")
            config["driven_strip_min_x"] = 1.0
            path = Path(temp) / "bad-strip.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "two vertex columns"):
                pilot.read_config(path)


if __name__ == "__main__":
    unittest.main()
